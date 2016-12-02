import xml.etree.cElementTree as ET

root = ET.Element("experimentData")

instance = ET.SubElement(root, "dataset")
ET.SubElement(instance, "path").text = '/home/hmedal/Documents/2_msu/research_manager/data/facLoc/Daskin/Daskin88_FacPro_p4.xml'
ET.SubElement(instance, "name").text = 'd88'
ET.SubElement(instance, "numFacs").text = '4'

instance = ET.SubElement(root, "instance")
ET.SubElement(instance, "numCapLevels").text = "3"
ET.SubElement(instance, "numAllocLevels").text = "3"
ET.SubElement(instance, "budget").text = "2"

algorithm = ET.SubElement(root, "algorithm")
ET.SubElement(algorithm, "numSamplesExponent").text = "3"
ET.SubElement(algorithm, "numSamplesForFinalExponent").text = "3"
ET.SubElement(algorithm, "deltaExponent").text = "1.5"
ET.SubElement(algorithm, "excess_capacity").text = "0.3"
ET.SubElement(algorithm, "penaltyMultiplier").text = "0.5"
ET.SubElement(algorithm, "numThreads").text = "4"

tree = ET.ElementTree(root)
path1 ='/home/hmedal/Documents/2_msu/1_MSU_Projects/facilityProtection_CSF/runFiles/random-draw/parameterSettings.xml'
path2 ='/home/hmedal/Documents/2_msu/1_MSU_Projects/facilityProtection_CSF/runFiles/imperfectPro-submod/parameterSettings.xml'
tree.write(path1)
tree.write(path2)