from pylab import *

x = arange(0.0, 1.0, 0.01)
numHazardLevels = 3
y1 = x**((0.0)/numHazardLevels)
y2 = x**((1.0)/numHazardLevels)
y3 = x**((2.0)/numHazardLevels)
y4 = x**((3.0)/numHazardLevels)
print y4
fig, ax = plt.subplots()
ax.plot(x, y1, label = 'y1 ' + str((0.0)/numHazardLevels))
ax.plot(x, y2, label = 'y2 ' + str((1.0)/numHazardLevels))
ax.plot(x, y3, label = 'y3 ' + str((2.0)/numHazardLevels))
ax.plot(x, y4, label = 'y4 ' + str((3.0)/numHazardLevels))
xlabel('allocation')
ylabel('probability')
title('About as simple as it gets, folks')
# Now add the legend with some customizations.
legend = ax.legend(loc='upper center', shadow=True)
grid(True)
#savefig("test.png")
show()