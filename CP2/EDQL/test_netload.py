import sumolib

print("👣 Trying to read cp2.net.xml...")
net = sumolib.net.readNet("cp2.net.xml")
print("✅ Successfully loaded net with", len(net.getEdges()), "edges")
