import xml.etree.cElementTree as ET
import math
from edu.msstate.hm568.impro import executableModel, imperfectPro_dataset,\
    imperfectPro_model, imperfectPro_problemInstance
from edu.msstate.hm568.impro.imperfectPro_problemInstance import Instance
import csv

EARTH_RADIUS = 6378137     # earth circumference in meters

def great_circle_distance(latlong_a, latlong_b):
    lat1, lon1 = latlong_a
    lat2, lon2 = latlong_b
 
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    a = (math.sin(dLat / 2) * math.sin(dLat / 2) +
            math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * 
            math.sin(dLon / 2) * math.sin(dLon / 2))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d_meters = EARTH_RADIUS * c
    d_miles = (1.0/1000)*0.621371*d_meters
    return d_miles

def getHazardLevelForLocationInScenario(coorOfHazardCenter, coorOfLoc, severityRadii):
    distFromCenterToLocation = great_circle_distance(coorOfHazardCenter, coorOfLoc)
    hazardLevel = 0
    numLevels = len(severityRadii)
    #print "distFromCenterToLocation", coorOfHazardCenter, coorOfLoc, distFromCenterToLocation
    for level in reversed(range(numLevels)):
        if(distFromCenterToLocation <= severityRadii[level]):
            hazardLevel = numLevels - level
    #print "hazardLevel", hazardLevel
    return hazardLevel
            
    
def setPaths():
    global dataPath
    dataPath = '/home/hmedal/Documents/2_msu/research_manager/data/'

def readFromCSVFile(filepath, numLevels):
    centers = []
    severityRadii = []
    probabilities = []
    with open(filepath, 'rb') as csvfile:
        myReader = csv.reader(csvfile, delimiter=',', quotechar='|')
        myReader.next()
        for row in myReader:
            index = 4
            latIndex = index
            index += 1
            lngIndex = index
            index += 1
            centers.append([float(row[latIndex]), float(row[lngIndex])])
            levelsRadii = []
            for levelIndex in range(numLevels):
                levelsRadii.append(float(row[index]))
                index += 1
            severityRadii.append(levelsRadii)
            probabilities.append(float(row[index]))
            #print ','.join(row)
    #print centers
    #print severityRadii
    #print probabilities
    return centers, severityRadii, probabilities
        
def createHazardsFile_allFullyExposedAlways(numFacs, numHazardLevels):
    endingStr = "_allFullyExposedAlways"
    root = ET.Element("hazardsData")
    numHazardLevelsElement = ET.SubElement(root, "numHazardLevels")
    numHazardLevelsElement.text = str(numHazardLevels)
    scenariosElement = ET.SubElement(root, "scenarios")
    for scenIndex in range(1):
        scenarioElement = ET.SubElement(scenariosElement, "scenario")
        scenarioElement.set('id', str(scenIndex))
        ET.SubElement(scenarioElement, "prob").text = str(1.0)
        for fac in range(numFacs):
            facElement = ET.SubElement(scenarioElement, "hazardLevelAtFacility")
            facElement.set('facName', str(fac))
            facElement.text = str(numHazardLevels - 1)
    
    tree = ET.ElementTree(root)
    filename = dataPath + '/facLoc/Daskin/Hazards/hazardsDef_custom_facs' +str(numFacs) + '_levels' +str(numHazardLevels) + endingStr + '.xml'
    tree.write(filename)
    
def createHazardsFile_HalfExposedAlways(numFacs, numHazardLevels):
    endingStr = "_halfExposedAlways"
    root = ET.Element("hazardsData")
    numHazardLevelsElement = ET.SubElement(root, "numHazardLevels")
    numHazardLevelsElement.text = str(numHazardLevels)
    scenariosElement = ET.SubElement(root, "scenarios")
    for scenIndex in range(1):
        scenarioElement = ET.SubElement(scenariosElement, "scenario")
        scenarioElement.set('id', str(scenIndex))
        ET.SubElement(scenarioElement, "prob").text = str(1.0)
        for fac in range(numFacs):
            facElement = ET.SubElement(scenarioElement, "hazardLevelAtFacility")
            facElement.set('facName', str(fac))
            facElement.text = str(1)
    
    tree = ET.ElementTree(root)
    filename = dataPath + '/facLoc/Daskin/Hazards/hazardsDef_custom_facs' +str(numFacs) + '_levels' +str(numHazardLevels) + endingStr + '.xml'
    tree.write(filename)

#conditional: if True, probabilities are normalized; meaning that the normalized probabilities are conditional probabilities, given that a hazard occurs
#
#
def createHazardsFile_readFromFile(numFacs, hazardsFilepath, locCoor, numLevels, conditional = True):
    centers, severityRadii, probabilities = readFromCSVFile(hazardsFilepath, numLevels)
    endingStr = ""
    print "probabilities", probabilities
    if(conditional):
        sumOfProbs = sum(probabilities)
        probabilitiesToWrite = [prob/sumOfProbs for prob in probabilities]
        endingStr = "_conditional"
    else:
        probabilitiesToWrite = probabilities
    print "probabilitiesToWrite", probabilitiesToWrite
    numScenarios = len(centers)
    numHazardLevels = len(severityRadii[0])+1
    root = ET.Element("hazardsData")
    numHazardLevelsElement = ET.SubElement(root, "numHazardLevels") #includes the zero level
    numHazardLevelsElement.text = str(numHazardLevels)
    scenariosElement = ET.SubElement(root, "scenarios")
    for scenIndex in range(numScenarios):
        scenarioElement = ET.SubElement(scenariosElement, "scenario")
        scenarioElement.set('id', str(scenIndex))
        ET.SubElement(scenarioElement, "prob").text = str(probabilitiesToWrite[scenIndex])
        for fac in range(numFacs):
            facElement = ET.SubElement(scenarioElement, "hazardLevelAtFacility")
            facElement.set('facName', str(fac))
            facElement.text = str(getHazardLevelForLocationInScenario(centers[scenIndex], locCoor[fac], severityRadii[scenIndex]))
    if(not conditional):
        scenarioElementLast = ET.SubElement(scenariosElement, "scenario")
        scenarioElementLast.set('id', str(numScenarios))
        ET.SubElement(scenarioElementLast, "prob").text = str(1-sum(probabilities))
        for fac in range(numFacs):
            facElement = ET.SubElement(scenarioElementLast, "hazardLevelAtFacility")
            facElement.set('facName', str(fac))
            facElement.text = str(0)
    tree = ET.ElementTree(root)
    filename = dataPath + '/facLoc/Daskin/Hazards/hazardsDef_custom_facs' +str(numFacs) + '_levels' +str(numHazardLevels) + endingStr + '.xml'
    tree.write(filename)
    
def test_GC_dist():
    coord_pairs = [
         # between eighth and 31st and eighth and 30th
         [(40.750307,-73.994819), (40.749641,-73.99527)],
         # sanfran to NYC ~2568 miles
         [(37.784750,-122.421180), (40.714585,-74.007202)],
         # about 10 feet apart
         [(40.714732,-74.008091), (40.714753,-74.008074)],
         # inches apart
         [(40.754850,-73.975560), (40.754851,-73.975561)],
    ]
    for pair in coord_pairs:
        miles = great_circle_distance(pair[0], pair[1]) # doctest: +ELLIPSIS
        print "miles", miles

#test_GC_dist() 
setPaths()
hazardsFilePath = dataPath + '/facLoc/Daskin/Hazards/list_of_hazard_scenarios_2_levels.csv'

for p in [2,3,4,5,6,7,8,9,10]:
    dataset = imperfectPro_dataset.ImproDataset()
    dataset.readInDataset(dataPath + '/facLoc/Daskin/Daskin49_FacPro_p' + str(p) + '.xml')
    locCoor = dataset.coor
    numFacs = len(locCoor)
    #readFromCSVFile(filename)
    createHazardsFile_readFromFile(numFacs, hazardsFilePath, locCoor, 2, True)
    createHazardsFile_allFullyExposedAlways(numFacs, 3)
    createHazardsFile_HalfExposedAlways(numFacs, 3)
print "completed"
#createHazardsFile_allFullyExposedAlways(49, 2)