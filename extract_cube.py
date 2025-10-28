import numpy as np
from scipy.spatial import cKDTree
import os

class MeshExtractor:
    def __init__(self, vtk_file):
        self.vtk_file = vtk_file
        self.nodes = {}
        self.elements = {}
        self.material = {}
        
    def read_vtk_file(self):
        """Read the .vtk file and extract nodes, elements, and material IDs"""
        if not os.path.exists(self.vtk_file):
            raise FileNotFoundError(f"Input file '{self.vtk_file}' not found")
            
        print(f"Reading VTK file: {self.vtk_file}")
        
        with open(self.vtk_file, 'r') as f:
            lines = f.readlines()
        
        # Parse VTK file
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            if not line or line.startswith('#'):
                i += 1
                continue
            
            # Look for POINTS section
            if line.startswith('POINTS'):
                parts = line.split()
                num_points = int(parts[1])
                i += 1
                self._read_points(lines, i, num_points)
                i += (num_points * 3 + 2) // 3
                continue
                
            # Look for CELLS section
            elif line.startswith('CELLS'):
                parts = line.split()
                num_cells = int(parts[1])
                i += 1
                self._read_cells(lines, i, num_cells)
                i += num_cells
                continue
                
            # Look for CELL_TYPES section
            elif line.startswith('CELL_TYPES'):
                parts = line.split()
                num_cell_types = int(parts[1])
                i += 1
                i += (num_cell_types + 9) // 10
                continue
                
            # Look for CELL_DATA section
            elif line.startswith('CELL_DATA'):
                parts = line.split()
                num_cells = int(parts[1])
                i += 1
                while i < len(lines):
                    data_line = lines[i].strip()
                    if data_line.startswith('SCALARS') and 'material' in data_line.lower():
                        i += 1
                        if i < len(lines) and lines[i].strip().startswith('LOOKUP_TABLE'):
                            i += 1
                        self._read_material_data(lines, i, num_cells)
                        break
                    i += 1
                break
                
            i += 1
                
    def _read_points(self, lines, start_idx, num_points):
        """Read point coordinates from VTK file"""
        points_data = []
        line_idx = start_idx
        
        while len(points_data) < num_points * 3 and line_idx < len(lines):
            line = lines[line_idx].strip()
            if line:
                coords = line.split()
                points_data.extend([float(x) for x in coords])
            line_idx += 1
        
        for i in range(num_points):
            node_id = i + 1
            idx = i * 3
            if idx + 2 < len(points_data):
                self.nodes[node_id] = np.array([
                    points_data[idx], 
                    points_data[idx + 1], 
                    points_data[idx + 2]
                ])
    
    def _read_cells(self, lines, start_idx, num_cells):
        """Read cell connectivity from VTK file"""
        cell_idx = 0
        line_idx = start_idx
        
        while cell_idx < num_cells and line_idx < len(lines):
            line = lines[line_idx].strip()
            if line:
                parts = [int(x) for x in line.split()]
                num_nodes = parts[0]
                node_ids = [x + 1 for x in parts[1:num_nodes + 1]]
                
                elem_id = cell_idx + 1
                self.elements[elem_id] = node_ids
                cell_idx += 1
                
            line_idx += 1
    
    def _read_material_data(self, lines, start_idx, num_cells):
        """Read material data from VTK file"""
        materials_data = []
        line_idx = start_idx
        
        while len(materials_data) < num_cells and line_idx < len(lines):
            line = lines[line_idx].strip()
            if line:
                materials = line.split()
                materials_data.extend([int(x) for x in materials])
            line_idx += 1
        
        for i, elem_id in enumerate(sorted(self.elements.keys())):
            if i < len(materials_data):
                self.material[elem_id] = materials_data[i]
            else:
                self.material[elem_id] = 1
        
        if materials_data:
            unique_mats = set(materials_data)
            print(f"Material IDs in input .vtk file: {sorted(unique_mats)}")
    
    def get_mesh_bounds(self):
        """Get the bounding box of the mesh"""
        if not self.nodes:
            print("Warning: No nodes found")
            return None
            
        coords = np.array(list(self.nodes.values()))
        min_coords = np.min(coords, axis=0)
        max_coords = np.max(coords, axis=0)
        center = (min_coords + max_coords) / 2
        
        return {
            'min': min_coords,
            'max': max_coords,
            'center': center,
            'size': max_coords - min_coords
        }
    
    def extract_cube_region(self, cube_dims, center=None):
        """Extract elements that intersect with the cube region"""
        bounds = self.get_mesh_bounds()
        if bounds is None:
            raise ValueError("Cannot extract cube region: no mesh data available")
            
        if center is None:
            center = bounds['center']
        
        print(f"Using cube center: {center}")
        print(f"Cube dimensions: {cube_dims}")
        
        half_dims = np.array(cube_dims) / 2
        cube_min = center - half_dims
        cube_max = center + half_dims
        
        print(f"Cube bounds: {cube_min} to {cube_max}")
        
        intersecting_elements = {}
        intersecting_nodes = set()
        
        for elem_id, node_ids in self.elements.items():
            try:
                elem_coords = np.array([self.nodes[nid] for nid in node_ids])
                elem_min = np.min(elem_coords, axis=0)
                elem_max = np.max(elem_coords, axis=0)
                
                if (np.all(elem_min <= cube_max) and np.all(elem_max >= cube_min)):
                    intersecting_elements[elem_id] = node_ids
                    intersecting_nodes.update(node_ids)
            except KeyError as e:
                print(f"Warning: Element {elem_id} references missing node {e}")
                continue
        
        return intersecting_elements, intersecting_nodes, (cube_min, cube_max)
    
    def create_hexahedral_mesh(self, cube_bounds, n_elements):
        """Create a structured hexahedral mesh within the cube"""
        cube_min, cube_max = cube_bounds
        n1, n2, n3 = n_elements
        
        print(f"Creating {n1}x{n2}x{n3} structured mesh")
        
        x = np.linspace(cube_min[0], cube_max[0], n1 + 1)
        y = np.linspace(cube_min[1], cube_max[1], n2 + 1)
        z = np.linspace(cube_min[2], cube_max[2], n3 + 1)
        
        # Generate nodes
        new_nodes = {}
        node_counter = 1
        
        for k in range(n3 + 1):
            for j in range(n2 + 1):
                for i in range(n1 + 1):
                    new_nodes[node_counter] = np.array([x[i], y[j], z[k]])
                    node_counter += 1
        
        # Generate hexahedral elements
        new_elements = {}
        new_material = {}
        elem_counter = 1
        
        # Build kdtree for material interpolation
        kdtree = None
        elem_mats = []
        
        if self.nodes and self.material:
            print("Building KDTree for material interpolation...")
            elem_centroids = []
            elem_mats = []
            for eid, node_ids in self.elements.items():
                try:
                    centroid = np.mean([self.nodes[nid] for nid in node_ids], axis=0)
                    elem_centroids.append(centroid)
                    elem_mats.append(self.material.get(eid, 1))
                except KeyError:
                    continue
            
            if elem_centroids:
                kdtree = cKDTree(elem_centroids)
        
        for k in range(n3):
            for j in range(n2):
                for i in range(n1):
                    n1_idx = k * (n2 + 1) * (n1 + 1) + j * (n1 + 1) + i + 1
                    n2_idx = n1_idx + 1
                    n3_idx = n1_idx + (n1 + 1) + 1
                    n4_idx = n1_idx + (n1 + 1)
                    n5_idx = n1_idx + (n2 + 1) * (n1 + 1)
                    n6_idx = n5_idx + 1
                    n7_idx = n5_idx + (n1 + 1) + 1
                    n8_idx = n5_idx + (n1 + 1)
                    
                    element_nodes = [n1_idx, n2_idx, n3_idx, n4_idx, 
                                   n5_idx, n6_idx, n7_idx, n8_idx]
                    new_elements[elem_counter] = element_nodes
                    
                    if kdtree is not None:
                        elem_center = np.array([
                            (x[i] + x[i+1]) / 2,
                            (y[j] + y[j+1]) / 2,
                            (z[k] + z[k+1]) / 2
                        ])
                        _, nearest_idx = kdtree.query(elem_center)
                        new_material[elem_counter] = elem_mats[nearest_idx]
                    else:
                        new_material[elem_counter] = 1
                    
                    elem_counter += 1
        
        return new_nodes, new_elements, new_material

    def write_ucd_inp(self, output_file, nodes, elements, material,
                      boundary_elements=None, renumber_nodes_zero_based=True):
        """Write UCD/AVS-style .inp compatible with deal.II GridIn"""
        
        node_ids_sorted = sorted(nodes.keys())
        elem_ids_sorted = sorted(elements.keys())

        if renumber_nodes_zero_based:
            node_old2new = {nid: i for i, nid in enumerate(node_ids_sorted)}
        else:
            node_old2new = {nid: nid for nid in node_ids_sorted}

        num_nodes = len(node_ids_sorted)
        num_hex = len(elem_ids_sorted)
        num_quads = len(boundary_elements) if boundary_elements else 0
        num_elements = num_hex + num_quads

        with open(output_file, "w") as f:
            # Summary row
            f.write("\t".join([str(num_nodes), str(num_elements), "0", "0", "0"]) + "\n")

            # Write nodes
            for old_nid in node_ids_sorted:
                new_nid = node_old2new[old_nid]
                x, y, z = nodes[old_nid]
                f.write(f"{new_nid}\t{float(x)}\t{float(y)}\t{float(z)}\n")

            # Write hex elements
            for old_eid in elem_ids_sorted:
                mat_id = int(material.get(old_eid, 1))
                conn = elements[old_eid]
                if len(conn) != 8:
                    raise ValueError(f"Element {old_eid} is not a hex")
                conn_mapped = [node_old2new[n] for n in conn]
                f.write(
                    f"{old_eid}\t{mat_id}\thex\t" +
                    "\t".join(str(n) for n in conn_mapped) + "\n"
                )

            # Write boundary quads
            if boundary_elements:
                for be_id in sorted(boundary_elements.keys()):
                    be_mat, be_nodes = boundary_elements[be_id]
                    if len(be_nodes) != 4:
                        raise ValueError(f"Boundary element {be_id} is not a quad")
                    be_conn_mapped = [node_old2new[n] for n in be_nodes]
                    f.write(
                        f"{be_id}\t{int(be_mat)}\tquad\t" +
                        "\t".join(str(n) for n in be_conn_mapped) + "\n"
                    )

    def write_vtk_file(self, output_file, nodes, elements, material):
        """Write the mesh to a VTK file"""
        with open(output_file, 'w') as f:
            f.write("# vtk DataFile Version 3.0\n")
            f.write("Extracted and remeshed cube\n")
            f.write("ASCII\n")
            f.write("DATASET UNSTRUCTURED_GRID\n")
            
            # Write points
            f.write("POINTS {} float\n".format(len(nodes)))
            for node_id in sorted(nodes.keys()):
                coord = nodes[node_id]
                f.write("{:.6e} {:.6e} {:.6e}\n".format(coord[0], coord[1], coord[2]))
            
            # Write cells
            total_cell_size = len(elements) * 9
            f.write("CELLS {} {}\n".format(len(elements), total_cell_size))
            for elem_id in sorted(elements.keys()):
                node_ids = elements[elem_id]
                vtk_nodes = [nid - 1 for nid in node_ids]
                f.write("8 {}\n".format(" ".join(map(str, vtk_nodes))))
            
            # Write cell types (12 = VTK_HEXAHEDRON)
            f.write("CELL_TYPES {}\n".format(len(elements)))
            for _ in range(len(elements)):
                f.write("12\n")
            
            # Write material IDs
            f.write("CELL_DATA {}\n".format(len(elements)))
            f.write("SCALARS MaterialID int 1\n")
            f.write("LOOKUP_TABLE default\n")
            for elem_id in sorted(elements.keys()):
                mat_id = material.get(elem_id, 1)
                f.write("{}\n".format(mat_id))

    def build_left_boundary_quads(self, n_elements, new_elements, material_id=400):
        """Build QUAD faces on the left (min-x) plane"""
        n1, n2, n3 = n_elements

        stride_i = 1
        stride_j = (n1 + 1)
        stride_k = (n2 + 1) * (n1 + 1)

        def nid(i, j, k):
            return k * stride_k + j * stride_j + i * stride_i + 1

        quad_id = max(new_elements.keys()) + 1
        boundary_elements = {}

        for k in range(n3):
            for j in range(n2):
                n0 = nid(0, j, k)
                n1q = nid(0, j + 1, k)
                n2q = nid(0, j + 1, k + 1)
                n3q = nid(0, j, k + 1)
                boundary_elements[quad_id] = (material_id, [n0, n1q, n2q, n3q])
                quad_id += 1

        return boundary_elements

    def process_cube_extraction(self, cube_dims, n_elements, center=None, output_filename=None):
        """Main function to extract and remesh cube from VTK"""
        self.read_vtk_file()
        
        if len(self.nodes) == 0:
            print("ERROR: No nodes were read from the VTK file!")
            return None, None, None
            
        print("Extracting cube region...")
        intersecting_elements, intersecting_nodes, cube_bounds = self.extract_cube_region(
            cube_dims, center)
        print(f"Found {len(intersecting_elements)} intersecting elements")
        
        print("Creating hexahedral mesh...")
        new_nodes, new_elements, new_material = self.create_hexahedral_mesh(
            cube_bounds, n_elements)
        print(f"Created {len(new_nodes)} nodes and {len(new_elements)} elements")

        # Build left-face boundary quads
        left_quads = self.build_left_boundary_quads(n_elements, new_elements, material_id=400)
      
        # Write UCD .inp file
        ucd_file = output_filename + "_UCD.inp"
        print(f"Writing UCD .inp file: {ucd_file}")
        self.write_ucd_inp(
            ucd_file,
            new_nodes,
            new_elements,
            new_material,
            boundary_elements=left_quads,
            renumber_nodes_zero_based=True
        )
        
        # Write VTK file
        vtk_file = output_filename + '.vtk'
        print(f"Writing VTK file: {vtk_file}")
        self.write_vtk_file(vtk_file, new_nodes, new_elements, new_material)
        
        return new_nodes, new_elements, new_material

def main():
    input_file = "rampp_VTK_coarse.vtk"
    cube_dimensions = [12.0, 12.0, 12.0]
    n_elements = [60, 60, 60]
    cube_center = np.array([0, 0, 0])
    output_filename = "extracted_cube_coarse"
    
    # Create extractor and process
    extractor = MeshExtractor(input_file)
    nodes, elements, material = extractor.process_cube_extraction(
        cube_dimensions, n_elements, center=cube_center, output_filename=output_filename)

if __name__ == "__main__":
    main()