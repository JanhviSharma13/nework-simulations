#!/usr/bin/env python3
"""
Check edge IDs in cleaned network files
"""

import xml.etree.ElementTree as ET
import os

def check_network_edges(net_file: str, max_edges: int = 20):
    """Check edge IDs in a network file"""
    print(f"\nChecking edges in {net_file}")
    
    if not os.path.exists(net_file):
        print(f"File not found: {net_file}")
        return
    
    try:
        tree = ET.parse(net_file)
        root = tree.getroot()
        
        edges = root.findall('.//edge')
        print(f"Total edges: {len(edges)}")
        
        print("First 20 edge IDs:")
        for i, edge in enumerate(edges[:max_edges]):
            edge_id = edge.get('id')
            from_node = edge.get('from')
            to_node = edge.get('to')
            print(f"  {i+1}: {edge_id} ({from_node} -> {to_node})")
        
        if len(edges) > max_edges:
            print(f"  ... and {len(edges) - max_edges} more edges")
            
    except Exception as e:
        print(f"Error parsing {net_file}: {e}")

def main():
    """Check all cleaned network files"""
    networks = [
        'Rakab-Ganj/EDQL-RG/EDQL-CP2/cp2_cleaned.net.xml',
        'Rakab-Ganj/EDQL-RG/EDQL-CP/cp_cleaned.net.xml',
        'Rakab-Ganj/EDQL-RG/EDQL-RakabGanj/rakabganj_cleaned.net.xml',
        'Safdarjung/safdarjung_cleaned.net.xml',
        'Chandni-Chowk/chandnichowk_cleaned.net.xml'
    ]
    
    for net_file in networks:
        check_network_edges(net_file)

if __name__ == "__main__":
    main() 