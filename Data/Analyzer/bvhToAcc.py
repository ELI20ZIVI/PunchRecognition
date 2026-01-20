import numpy as np
import pandas as pd
import re

class BVHJoint:
    def __init__(self, name, parent=None):
        self.name = name
        self.parent = parent
        self.children = []
        self.offset = np.zeros(3)
        self.channels = []
        self.channel_indices = []  # Indices in the motion data array
        self.matrix_local = np.identity(4)
        self.world_pos = np.zeros(3)

def euler_to_matrix(channels, values, order="ZXY"):
    # Costruisce la matrice di rotazione dai canali Euler (gradi)
    # Ordine standard BVH spesso è Zrotation, Xrotation, Yrotation
    rx, ry, rz = 0, 0, 0
    for ch, val in zip(channels, values):
        if ch.lower() == 'xrotation': rx = np.radians(val)
        elif ch.lower() == 'yrotation': ry = np.radians(val)
        elif ch.lower() == 'zrotation': rz = np.radians(val)
    
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    
    # Matrici base
    m_x = np.array([[1, 0, 0, 0], [0, cx, -sx, 0], [0, sx, cx, 0], [0, 0, 0, 1]])
    m_y = np.array([[cy, 0, sy, 0], [0, 1, 0, 0], [-sy, 0, cy, 0], [0, 0, 0, 1]])
    m_z = np.array([[cz, -sz, 0, 0], [sz, cz, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    
    # Moltiplicazione nell'ordine specificato (di solito Z * X * Y per BVH)
    # Nota: la moltiplicazione matriciale va letta da destra a sinistra rispetto all'ordine di applicazione
    # Se l'ordine nel file è Z, X, Y, la matrice risultante è Y @ X @ Z
    
    mat = np.identity(4)
    # Ordine tipico BVH: Z, poi X, poi Y
    mat = m_y @ m_x @ m_z
    return mat

def parse_bvh_hierarchy(filename):
    joints = {}
    joint_stack = []
    root_joint = None
    motion_data_start = 0
    channel_counter = 0
    frame_time = 0.016667

    with open(filename, 'r') as f:
        lines = f.readlines()
        
    for i, line in enumerate(lines):
        line = line.strip()
        tokens = re.split(r'\s+', line)
        
        if tokens[0] == "ROOT" or tokens[0] == "JOINT":
            name = tokens[1]
            parent = joint_stack[-1] if joint_stack else None
            joint = BVHJoint(name, parent)
            joints[name] = joint
            if parent: parent.children.append(joint)
            if not root_joint: root_joint = joint
            joint_stack.append(joint)
            
        elif tokens[0] == "End": # End Site
            joint = BVHJoint("EndSite_" + joint_stack[-1].name, joint_stack[-1])
            joint_stack[-1].children.append(joint)
            joint_stack.append(joint)
            
        elif tokens[0] == "OFFSET":
            joint_stack[-1].offset = np.array([float(tokens[1]), float(tokens[2]), float(tokens[3])])
            
        elif tokens[0] == "CHANNELS":
            count = int(tokens[1])
            joint_stack[-1].channels = tokens[2:]
            joint_stack[-1].channel_indices = list(range(channel_counter, channel_counter + count))
            channel_counter += count
            
        elif tokens[0] == "}":
            joint_stack.pop()
            
        elif tokens[0] == "MOTION":
            pass
        elif tokens[0] == "Frame" and tokens[1] == "Time:":
            frame_time = float(tokens[2])
            motion_data_start = i + 1
            break
            
    return root_joint, joints, lines[motion_data_start:], frame_time

def calculate_accelerations(bvh_file, target_joints=["RightHand", "LeftHand"]):
    print(f"Elaborazione file: {bvh_file}...")
    root, all_joints, motion_lines, frame_time = parse_bvh_hierarchy(bvh_file)
    
    # Caricamento dati motion
    motion_data = np.loadtxt(motion_lines)
    num_frames = motion_data.shape[0]
    
    print(f"Frame totali: {num_frames}, Frame Time: {frame_time}s")
    
    # Dizionario per salvare le posizioni nel tempo
    trajectories = {name: [] for name in target_joints}
    times = []

    for f in range(num_frames):
        frame_vals = motion_data[f]
        times.append(f * frame_time)
        
        # Forward Kinematics
        # Stack per attraversare l'albero: (joint, parent_transform)
        stack = [(root, np.identity(4))]
        
        while stack:
            joint, parent_mat = stack.pop(0)
            
            # 1. Costruisci matrice locale (Traslazione Offset)
            local_trans = np.identity(4)
            local_trans[:3, 3] = joint.offset
            
            # 2. Aggiungi Rotazione (se ci sono canali)
            local_rot = np.identity(4)
            if joint.channel_indices:
                # Estrae i valori per questo giunto
                vals = frame_vals[joint.channel_indices]
                chans = joint.channels
                
                # Se è il Root, gestisci anche la posizione globale
                pos_offset = np.zeros(3)
                rot_vals = []
                rot_chans = []
                
                val_idx = 0
                for ch in chans:
                    if 'position' in ch.lower():
                        if 'X' in ch: pos_offset[0] = vals[val_idx]
                        if 'Y' in ch: pos_offset[1] = vals[val_idx]
                        if 'Z' in ch: pos_offset[2] = vals[val_idx]
                    elif 'rotation' in ch.lower():
                        rot_vals.append(vals[val_idx])
                        rot_chans.append(ch)
                    val_idx += 1
                
                # Applica traslazione del root se presente
                local_trans[:3, 3] += pos_offset
                
                # Calcola matrice di rotazione
                local_rot = euler_to_matrix(rot_chans, rot_vals)

            # Matrice Globale = Parent * (Offset + Pos) * Rotazione
            global_mat = parent_mat @ local_trans @ local_rot
            
            # Salva posizione se è un target
            if joint.name in target_joints:
                trajectories[joint.name].append(global_mat[:3, 3])
            
            # Continua con i figli
            for child in joint.children:
                stack.append((child, global_mat))

    # Calcolo Derivate e Salvataggio
    for joint_name in target_joints:
        pos_array = np.array(trajectories[joint_name]) # Shape (N, 3)
        
        # Velocità (m/s)
        vel = np.gradient(pos_array, frame_time, axis=0)
        # Accelerazione (m/s^2)
        acc = np.gradient(vel, frame_time, axis=0)
        # Modulo
        acc_mag = np.linalg.norm(acc, axis=1)
        
        # Converti in DataFrame
        df = pd.DataFrame({
            'Time_sec': times,
            'PosX': pos_array[:,0], 'PosY': pos_array[:,1], 'PosZ': pos_array[:,2],
            'AccX': acc[:,0], 'AccY': acc[:,1], 'AccZ': acc[:,2],
            'Acc_Total': acc_mag
        })
        
        # Opzionale: Converti cm -> metri se il file è in cm
        # df['Acc_Total_m_s2'] = df['Acc_Total'] / 100.0 
        
        csv_name = f"accelerazioni_{joint_name}.csv"
        df.to_csv(csv_name, index=False, sep=',')
        print(f"Salvato: {csv_name} (Picco accelerazione: {np.max(acc_mag):.2f})")

# --- ESECUZIONE ---
input_file = "../BVH-60Hz/Elia 2 - Take 2026-01-14 04.29.52 PM_002_Skeleton 001.bvh"

# Puoi cambiare i nomi qui se nel tuo file sono diversi (es. RightWrist)
calculate_accelerations(input_file, target_joints=["RightHand", "LeftHand"])