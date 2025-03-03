import os
import json
import re
import cv2
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

# import standalone file handling functions
from gen import edit_tex_file, compile_tex_to_pdf, convert_pdf_to_png

def parse_tikz_coordinates(content):
    """
    parse coordinates from tikz draw commands to preserve line connectivity
    
    args:
        content: string containing tikz commands
    
    returns:
        list of lists containing line coordinates
    """
    # find all draw commands that create lines
    draw_commands = re.findall(r'\\draw \[.*?\] (.*?);', content)
    lines = []
    
    for cmd in draw_commands:
        if 'circle' in cmd or 'node' in cmd:  # skip circle and node commands
            continue
            
        # split the command by '--' to get connected points in order
        segments = cmd.split('--')
        if segments:
            line_points = []
            for segment in segments:
                # extract coordinates, keeping only the first pair in each segment
                coords = re.findall(r'\(([-\d.]+),([-\d.]+)\)', segment)
                if coords:
                    # take only the first coordinate pair if multiple exist
                    x, y = coords[0]
                    line_points.append([float(x), float(y)])
            
            if line_points:  # only add if we found points
                lines.append(line_points)
    
    return lines

def extract_node_labels(content):
    """
    Extract node labels and their coordinates from the content
    
    args:
        content: string containing tikz commands
        
    returns:
        dictionary mapping coordinates to node labels
    """
    # Find all node commands
    node_commands = re.findall(r'\\node \[.*?\] at \(([-\d.]+),([-\d.]+)\) \{(.*?)\};', content)
    
    # Create a dictionary mapping coordinates to labels
    coord_to_label = {}
    for x, y, label in node_commands:
        coord_to_label[(float(x), float(y))] = label.strip()
    
    return coord_to_label

def find_closest_node(coord, coord_to_label, threshold=0.01):
    """
    Find the closest node label to the given coordinate
    
    args:
        coord: (x, y) coordinate
        coord_to_label: dictionary mapping coordinates to node labels
        threshold: maximum distance to consider a match
        
    returns:
        node label or None if no close node is found
    """
    x, y = coord
    for (node_x, node_y), label in coord_to_label.items():
        distance = ((node_x - x)**2 + (node_y - y)**2)**0.5
        if distance < threshold:
            return label
    return None

def create_line_mask(start, end, image_size, line_width=3):
    """
    create a binary mask for a single line
    """
    mask = Image.new('L', image_size, 0)
    draw = ImageDraw.Draw(mask)
    draw.line([tuple(start), tuple(end)], fill=255, width=line_width)
    return mask

def normalize_coordinates(points, original_image):
    """
    normalize coordinates to match the original image dimensions and orientation
    
    args:
        points: list of [x, y] coordinates
        original_image: PIL Image object of the original generated image
    """
    points = np.array(points)
    width, height = original_image.size
    
    # flip y coordinates (tikz has y-up, PIL has y-down)
    points[:, 1] = -points[:, 1]
    
    # get min and max coordinates
    min_coords = points.min(axis=0)
    max_coords = points.max(axis=0)
    
    # calculate scale to match the image dimensions
    # leave some padding (e.g., 10% of image size)
    padding = 0.15
    scale = min((1 - 2 * padding) * width / (max_coords[0] - min_coords[0]),
                (1 - 2 * padding) * height / (max_coords[1] - min_coords[1]))
    
    # scale points
    points = points * scale
    
    # center in image
    points_min = points.min(axis=0)
    points_max = points.max(axis=0)
    center_offset = [
        width/2 - (points_max[0] + points_min[0])/2,
        height/2 - (points_max[1] + points_min[1])/2
    ]
    points = points + center_offset
    
    return points.tolist()

def generate_line_masks(metadata_path, output_dir="mask_samples"):
    """
    generate individual binary masks for each line by processing one line at a time
    
    args:
        metadata_path: path to metadata.json
        output_dir: directory to save the masks
    """
    # load metadata
    with open(metadata_path, 'r') as f:
        samples = json.load(f)
    
    # create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for sample in tqdm(samples):
        sample_id = sample['id']
        content = sample['content']
        
        # Extract node labels and their coordinates
        coord_to_label = extract_node_labels(content)
        
        # find all draw commands
        draw_commands = re.findall(r'(\\draw .*?;)', content)

        masks_path = os.path.join(output_dir, "masks")
        overlays_path = os.path.join(output_dir, "overlays")
        indv_samples_path = os.path.join(output_dir, "indv_samples")        
        
        os.makedirs(masks_path, exist_ok=True)
        os.makedirs(overlays_path, exist_ok=True)
        os.makedirs(indv_samples_path, exist_ok=True)
        
        # process each draw command individually
        line_idx = 0
        for cmd_idx, cmd in enumerate(draw_commands):
            # skip node commands
            if 'node' in cmd:
                continue
                
            # For circle commands, just change the color as before
            if 'circle' in cmd:
                # Extract circle center and radius
                circle_match = re.search(r'\(([-\d.]+),([-\d.]+)\) circle \(([-\d.]+)\)', cmd)
                if circle_match:
                    center_x, center_y, radius = map(float, circle_match.groups())
                    center_coord = (center_x, center_y)
                    
                    # Find the center node label if it exists
                    center_label = find_closest_node(center_coord, coord_to_label)
                    
                    # Create a simple highlight info for this circle
                    highlight_info = {
                        "type": "circle",
                        "center": {"x": center_x, "y": center_y},
                        "radius": radius
                    }
                    
                    if center_label:
                        highlight_info["center_label"] = center_label
                
                # Create modified content where only this circle is red
                # Since circle commands might not have 'black' explicitly, we need to handle it differently
                if 'black' in cmd:
                    modified_cmd = cmd.replace('black', 'red')
                else:
                    # If 'black' is not in the command, we need to insert the color
                    modified_cmd = cmd.replace('\\draw ', '\\draw [red] ')
                
                modified_content = content.replace(cmd, modified_cmd)
                
                # Make sure other draw commands stay black
                for other_cmd in draw_commands:
                    if other_cmd != cmd and 'node' not in other_cmd:
                        if 'red' in other_cmd:
                            if '[red]' in other_cmd:
                                other_modified = other_cmd.replace('[red]', '[black]')
                            else:
                                other_modified = other_cmd.replace('red', 'black')
                            modified_content = modified_content.replace(other_cmd, other_modified)
                
                # Process the modified content
                process_modified_content(modified_content, sample_id, line_idx, output_dir, indv_samples_path, masks_path, highlight_info)
                line_idx += 1
            else:
                # For line/polygon commands, identify individual edges
                # Extract the vertices from the command
                vertices_pattern = r'\(([-\d.]+),([-\d.]+)\)'
                vertices = re.findall(vertices_pattern, cmd)
                
                # Check if we have a polygon with multiple edges
                if len(vertices) >= 2:
                    # For each edge in the polygon
                    for i in range(len(vertices)):
                        # Get current vertex and next vertex (cycle back to first for the last edge)
                        current_vertex = vertices[i]
                        next_vertex = vertices[(i+1) % len(vertices)]
                        
                        # Find labels for the vertices if they exist
                        current_label = find_closest_node((float(current_vertex[0]), float(current_vertex[1])), coord_to_label)
                        next_label = find_closest_node((float(next_vertex[0]), float(next_vertex[1])), coord_to_label)
                        
                        # Create highlight info for this edge
                        highlight_info = {
                            "type": "edge",
                            "start": {"x": float(current_vertex[0]), "y": float(current_vertex[1])},
                            "end": {"x": float(next_vertex[0]), "y": float(next_vertex[1])}
                        }
                        
                        if current_label and next_label:
                            highlight_info["edge_name"] = f"line {current_label}{next_label}"
                        
                        # Create a copy of the original command
                        edge_cmd = cmd
                        
                        # Replace the original command with one where only this edge is red
                        # We'll create a new draw command for each edge
                        edge_content = content
                        
                        # Remove the original command from the content
                        edge_content = edge_content.replace(cmd, "")
                        
                        # Create individual draw commands for each edge
                        for j in range(len(vertices)):
                            v1 = vertices[j]
                            v2 = vertices[(j+1) % len(vertices)]
                            
                            # Determine color for this edge
                            color = "red" if j == i else "black"
                            
                            # Create a draw command for this edge
                            edge_draw = f"\\draw [thick, {color}] ({v1[0]},{v1[1]}) -- ({v2[0]},{v2[1]});\n"
                            edge_content = edge_content + edge_draw
                        
                        # Process the modified content
                        process_modified_content(edge_content, sample_id, line_idx, output_dir, indv_samples_path, masks_path, highlight_info)
                        line_idx += 1
        
        # create verification overlay with all masks
        create_overlay_visualization(sample, output_dir, masks_path, overlays_path)

def process_modified_content(modified_content, sample_id, line_idx, output_dir, indv_samples_path, masks_path, highlight_info=None):
    """
    Process modified content to generate masks
    
    args:
        modified_content: modified tex content
        sample_id: sample identifier
        line_idx: line index
        output_dir: output directory
        indv_samples_path: path for individual samples
        masks_path: path for masks
        highlight_info: dictionary containing information about the highlighted element
    """
    # save modified tex content
    tex_path = os.path.join(output_dir, f"mask_{sample_id}_{line_idx}.tex")
    edit_tex_file(tex_path, modified_content)
    
    # compile to pdf
    compile_tex_to_pdf(tex_path)
    os.system(f"mv mask_{sample_id}_{line_idx}.* {output_dir}/")
    
    # convert to png
    pdf_path = tex_path.replace('.tex', '.pdf')
    convert_pdf_to_png(pdf_path, output_dir, f"{sample_id}_{line_idx}", prefix="mask_")
    
    # clean up temporary files
    files = [f"mask_{sample_id}_{line_idx}.pdf", f"mask_{sample_id}_{line_idx}.aux", f"mask_{sample_id}_{line_idx}.tex", f"mask_{sample_id}_{line_idx}.log"]
    for f in files:
        fpath = os.path.join(output_dir, f)
        os.system(f"rm {fpath}")

    # create binary mask using color thresholding
    img = cv2.imread(os.path.join(output_dir, f"mask_page_{sample_id}_{line_idx}.png"))
    if img is None:
        print(f"warning: failed to read image for sample {sample_id}, line {line_idx}")
        return
    else:
        os.system(f"mv {output_dir}/mask_page_{sample_id}_{line_idx}.png {indv_samples_path}/")
        
    # extract red components (BGR format)
    red_mask = img[:,:,2] - img[:,:,1]//2 - img[:,:,0]//2
    
    # threshold to get binary mask
    _, binary_mask = cv2.threshold(red_mask, 50, 255, cv2.THRESH_BINARY)
    
    # save binary mask
    mask_path = os.path.join(masks_path, f"sample_{sample_id}_line_{line_idx}.png")
    cv2.imwrite(mask_path, binary_mask)
    
    # save highlight information to a JSON file if provided
    if highlight_info:
        # Add sample and line identifiers to the info
        highlight_info["sample_id"] = sample_id
        highlight_info["line_idx"] = line_idx
        
        info_path = os.path.join(masks_path, f"sample_{sample_id}_line_{line_idx}_info.json")
        with open(info_path, 'w') as f:
            json.dump(highlight_info, f, indent=2)

def create_overlay_visualization(sample, output_dir, masks_path, overlays_path):
    """
    create a visualization with the original image and all line masks overlaid
    
    args:
        sample: sample data from metadata
        output_dir: directory to save the output image
    """
    sample_id = sample['id']
    original_path = os.path.join("generated_samples", sample['image_path'])
    
    # load original image
    original = cv2.imread(original_path)
    if original is None:
        print(f"warning: failed to read original image {original_path}")
        return
        
    # create rgb overlay
    overlay = original.copy()
    
    # find all line masks for this sample
    line_idx = 0
    while True:
        mask_path = os.path.join(masks_path, f"sample_{sample_id}_line_{line_idx}.png")
        if not os.path.exists(mask_path):
            break
            
        # load mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # resize mask if needed
        if original.shape[:2] != mask.shape[:2]:
            mask = cv2.resize(mask, (original.shape[1], original.shape[0]))
        
        # overlay mask in red (BGR format)
        overlay[mask > 0] = [0, 0, 255]  # red in BGR
        
        line_idx += 1
    
    # save overlay
    overlay_path = os.path.join(overlays_path, f"sample_{sample_id}_overlay.png")
    cv2.imwrite(overlay_path, overlay)

if __name__ == '__main__':
    metadata_path = "generated_samples/metadata.json"
    generate_line_masks(metadata_path)
