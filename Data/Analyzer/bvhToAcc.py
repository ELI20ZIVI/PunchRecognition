import os
from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import gaussian_filter1d
import sys

# --- CLASS & PARSING (Unchanged) ---
class BvhJoint:
    def __init__(self, name, parent):
        self.name = name
        self.parent = parent
        self.offset = np.zeros(3)
        self.channel_names = []
        self.children = []
        
    def add_child(self, child):
        self.children.append(child)

def parse_bvh(file_path):
    with open(file_path, 'r') as f:
        content = f.read()

    # Split into Hierarchy and Motion
    hierarchy_str, motion_str = content.split('MOTION')
    
    # --- Parse Hierarchy ---
    lines = [l.strip() for l in hierarchy_str.split('\n') if l.strip()]
    
    joints = []
    stack = []
    
    for line in lines:
        parts = line.split()
        token = parts[0]
        
        if token in ['ROOT', 'JOINT']:
            name = parts[1]
            parent = stack[-1] if stack else None
            new_joint = BvhJoint(name, parent)
            joints.append(new_joint)
            if parent:
                parent.add_child(new_joint)
            stack.append(new_joint)
            
        elif token == 'End': # End Site
            name = stack[-1].name + "_EndSite"
            parent = stack[-1]
            new_joint = BvhJoint(name, parent)
            stack.append(new_joint)
            
        elif token == 'OFFSET':
            stack[-1].offset = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
            
        elif token == 'CHANNELS':
            count = int(parts[1])
            stack[-1].channel_names = parts[2 : 2+count]
            
        elif token == '}':
            stack.pop()

    # --- Parse Motion ---
    motion_lines = [l.strip() for l in motion_str.split('\n') if l.strip()]
    
    frames = 0
    frame_time = 0.0333
    motion_data = []
    
    for line in motion_lines:
        if line.startswith('Frames:'):
            frames = int(line.split()[1])
        elif line.startswith('Frame Time:'):
            # Some BVH exports use comma as decimal separator; normalize it.
            frame_str = line.split()[2].replace(',', '.')
            frame_time = float(frame_str)
        else:
            line_data = [float(x) for x in line.split()]
            motion_data.append(line_data)
            
    motion_arr = np.array(motion_data)
    
    return joints, motion_arr, frame_time

# --- CALCULATION (Unchanged) ---
def compute_fk_and_acceleration(joints, motion_data, frame_time):
    num_frames = motion_data.shape[0]
    num_joints = len(joints)
    
    # Map columns
    col_idx = 0
    joint_col_map = {}
    for joint in joints:
        num_ch = len(joint.channel_names)
        if num_ch > 0:
            joint_col_map[joint.name] = (col_idx, col_idx + num_ch)
            col_idx += num_ch
        else:
            joint_col_map[joint.name] = None

    world_positions = np.zeros((num_frames, num_joints, 3))
    print(f"Calculating positions for {num_frames} frames...")
    
    for f in range(num_frames):
        joint_matrices = {None: np.eye(4)} 
        
        for i, joint in enumerate(joints):
            local_translation = np.eye(4)
            local_translation[:3, 3] = joint.offset
            
            motion_transform = np.eye(4)
            
            if joint.name in joint_col_map and joint_col_map[joint.name] is not None:
                start, end = joint_col_map[joint.name]
                vals = motion_data[f, start:end]
                ch_names = joint.channel_names
                
                trans_vec = np.zeros(3)
                if 'Xposition' in ch_names: trans_vec[0] = vals[ch_names.index('Xposition')]
                if 'Yposition' in ch_names: trans_vec[1] = vals[ch_names.index('Yposition')]
                if 'Zposition' in ch_names: trans_vec[2] = vals[ch_names.index('Zposition')]
                
                motion_translation = np.eye(4)
                motion_translation[:3, 3] = trans_vec
                
                rot_channels = [c for c in ch_names if 'rotation' in c]
                if rot_channels:
                    euler_order = "".join([c[0] for c in rot_channels]).upper() 
                    euler_vals = [vals[ch_names.index(c)] for c in rot_channels]
                    
                    r = R.from_euler(euler_order, euler_vals, degrees=True)
                    rot_matrix = np.eye(4)
                    rot_matrix[:3, :3] = r.as_matrix()
                    motion_transform = motion_translation @ rot_matrix
                else:
                    motion_transform = motion_translation

            local_matrix = local_translation @ motion_transform
            parent_matrix = joint_matrices[joint.parent.name] if joint.parent else np.eye(4)
            world_matrix = parent_matrix @ local_matrix
            joint_matrices[joint.name] = world_matrix
            world_positions[f, i] = world_matrix[:3, 3]

    print("Calculating derivatives (with smoothing)...")
    dt = frame_time
    velocity = np.gradient(world_positions, axis=0) / dt
    acceleration = np.gradient(velocity, axis=0) / dt
    acceleration = gaussian_filter1d(acceleration, sigma=2, axis=0)
    
    return acceleration

# --- MAIN (Modified for specific export) ---
def main(input_bvh, output_csv):
    print(f"Reading {input_bvh}...")
    try:
        joints, motion_data, frame_time = parse_bvh(input_bvh)
    except FileNotFoundError:
        print("Error: File not found.")
        return

    accel_data = compute_fk_and_acceleration(joints, motion_data, frame_time)
    
    frames = accel_data.shape[0]
    joint_names = [j.name for j in joints]
    
    # --- MODIFICATION START ---
    target_joints = ['RightHand', 'LeftHand', 'r_hand', 'l_hand']
    col_names = []
    data_list = []
    
    # Create a map to find joint indices quickly
    name_to_index = {name: i for i, name in enumerate(joint_names)}
    
    print(f"Extracting data for: {', '.join(target_joints)}")
    
    for target in target_joints:
        if target in name_to_index:
            idx = name_to_index[target]
            # Add columns for X, Y, Z
            col_names.extend([f"{target}_Accel_X", f"{target}_Accel_Y", f"{target}_Accel_Z"])
            # Extract data for this joint (all frames, this joint index, all coordinates)
            data_list.append(accel_data[:, idx, :])
        else:
            print(f"WARNING: Joint '{target}' not found in BVH file.")

    if not data_list:
        print("Error: No target joints found. CSV will not be created.")
        return

    # Stack the selected data horizontally
    flat_data = np.hstack(data_list)
    
    df = pd.DataFrame(flat_data, columns=col_names)
    df.insert(0, 'Time', np.arange(frames) * frame_time)
    df.insert(0, 'Frame', np.arange(frames))

    # Rename columns to match desired output
    rename_map = {
        'r_hand_Accel_X': 'RightHand_Accel_X',
        'r_hand_Accel_Y': 'RightHand_Accel_Y',
        'r_hand_Accel_Z': 'RightHand_Accel_Z',
        'l_hand_Accel_X': 'LeftHand_Accel_X',
        'l_hand_Accel_Y': 'LeftHand_Accel_Y',
        'l_hand_Accel_Z': 'LeftHand_Accel_Z'
    }
    df.rename(columns=rename_map, inplace=True)

    # --- MODIFICATION END ---
    
    df.to_csv(output_csv, index=False)
    print(f"Success! Filtered acceleration data saved to: {output_csv}")

def _process_file(args):
    # Worker wrapper for multiprocessing
    input_folder, input_file = args
    input_bvh = os.path.join(input_folder, input_file)
    output_csv = os.path.join("../ProcessedData", f"{os.path.splitext(input_file)[0]}_accel.csv")
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    main(input_bvh, output_csv)


if __name__ == "__main__":
    folders = ["../Session1/Optitrack", "../Session1/Mocopi"]   # ../Session2/Optitrack - 60Hz

    tasks = []
    for input_folder in folders:
        for input_file in os.listdir(input_folder):
            if input_file.endswith(".bvh"):
                tasks.append((input_folder, input_file))

    if not tasks:
        print("No BVH files found to process.")
        sys.exit(0)

    workers = min(cpu_count(), len(tasks))
    if workers > 1:
        print(f"Processing {len(tasks)} files with {workers} workers...")
        with Pool(processes=workers) as pool:
            pool.map(_process_file, tasks)
    else:
        print("Processing files sequentially (single worker)...")
        for task in tasks:
            _process_file(task)