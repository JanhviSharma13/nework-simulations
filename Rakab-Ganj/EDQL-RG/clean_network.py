import os
import subprocess
from lxml import etree

def clean_network():
    """Clean the Rakab-Ganj network by removing problematic edges"""
    print("Cleaning Rakab-Ganj network...")
    
    # Read the original network file
    tree = etree.parse('../rg.net.xml')
    root = tree.getroot()
    
    # Find and remove edges with negative IDs
    edges_to_remove = []
    for edge in root.findall('.//edge'):
        edge_id = edge.get('id')
        if edge_id and edge_id.startswith('-'):
            edges_to_remove.append(edge)
            print(f"Marking for removal: {edge_id}")
    
    # Remove problematic edges
    for edge in edges_to_remove:
        parent = edge.getparent()
        if parent is not None:
            parent.remove(edge)
    
    print(f"Removed {len(edges_to_remove)} edges with negative IDs")
    
    # Save cleaned network
    cleaned_file = 'rg_cleaned.net.xml'
    tree.write(cleaned_file, encoding='utf-8', xml_declaration=True)
    print(f"Saved cleaned network to: {cleaned_file}")
    
    return cleaned_file

def generate_routes_for_cleaned_network(net_file):
    """Generate routes for the cleaned network"""
    print("Generating routes for cleaned network...")
    
    # Use randomTrips.py to generate routes
    cmd = [
        'python', 
        'C:/Users/Janhvi/Downloads/sumo/sumo-1.23.1/tools/randomTrips.py',
        '-n', net_file,
        '-r', 'rg_cleaned.rou.xml',
        '-b', '0',
        '-e', '100',
        '-p', '1.0',
        '--validate'
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print("Routes generated successfully!")
            return 'rg_cleaned.rou.xml'
        else:
            print(f"Error generating routes: {result.stderr}")
            return None
    except Exception as e:
        print(f"Error running randomTrips.py: {e}")
        return None

def test_cleaned_network():
    """Test the cleaned network"""
    print("Testing cleaned network...")
    
    try:
        import traci
        traci.start(['sumo', '-n', 'rg_cleaned.net.xml', '--no-warnings', '--no-step-log'])
        
        edges = traci.edge.getIDList()
        print(f"Total edges in cleaned network: {len(edges)}")
        
        # Check for negative edges
        neg_count = len([e for e in edges if e.startswith('-')])
        print(f"Negative edges remaining: {neg_count}")
        
        # Show sample edges
        print("Sample edge IDs:")
        for i, edge in enumerate(edges[:10]):
            print(f"  {i}: {edge}")
        
        traci.close()
        return neg_count == 0
        
    except Exception as e:
        print(f"Error testing cleaned network: {e}")
        return False

if __name__ == "__main__":
    # Clean the network
    cleaned_net = clean_network()
    
    # Test the cleaned network
    if test_cleaned_network():
        print("✅ Network cleaning successful!")
        
        # Generate routes
        route_file = generate_routes_for_cleaned_network(cleaned_net)
        if route_file:
            print(f"✅ Routes generated: {route_file}")
        else:
            print("❌ Failed to generate routes")
    else:
        print("❌ Network cleaning failed!") 