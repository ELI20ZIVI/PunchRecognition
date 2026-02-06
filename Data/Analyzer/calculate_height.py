#!/usr/bin/env python3
"""
Calculate subject height from BVH file - cleaner version
"""

def parse_bvh_simple(filepath):
    """Extract specific joint offsets from BVH file"""
    offsets = {}
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines):
        if 'MOTION' in line:
            break
        
        line_stripped = line.strip()
        
        # Extract joint offsets (both JOINT and ROOT)
        if line_stripped.startswith('JOINT ') or line_stripped.startswith('ROOT '):
            parts = line_stripped.split()
            joint_name = parts[1]
            
            # Look for OFFSET on the next few lines
            for j in range(i + 1, min(i + 5, len(lines))):
                if 'OFFSET' in lines[j]:
                    offset_parts = lines[j].strip().split()
                    if len(offset_parts) >= 4:
                        x, y, z = float(offset_parts[1]), float(offset_parts[2]), float(offset_parts[3])
                        offsets[joint_name] = (x, y, z)
                    break
        
        # Extract End Site offsets with their parent joint
        elif line_stripped == 'End Site':
            # Find the parent joint (look backward for the closest JOINT or ROOT)
            parent = None
            for j in range(i - 1, max(0, i - 10), -1):
                prev_line = lines[j].strip()
                if 'JOINT ' in prev_line or 'ROOT ' in prev_line:
                    parent = prev_line.split()[1]
                    break
            
            # Look for OFFSET in the next lines
            for j in range(i + 1, min(i + 5, len(lines))):
                if 'OFFSET' in lines[j]:
                    offset_parts = lines[j].strip().split()
                    if len(offset_parts) >= 4:
                        x, y, z = float(offset_parts[1]), float(offset_parts[2]), float(offset_parts[3])
                        if parent:
                            offsets[f'{parent}_EndSite'] = (x, y, z)
                    break
    
    return offsets

def calculate_chain_height(offsets):
    """Calculate height by summing the spine and leg chains"""
    
    # Head chain: sum Y offsets from Hips to Head End Site
    head_chain = ['Hips', 'Spine', 'Spine1', 'Neck', 'Head', 'Head_EndSite']
    head_y = 0.0
    for joint in head_chain:
        if joint in offsets:
            head_y += offsets[joint][1]  # Y is the second element
    
    # Leg chain: sum Y offsets from Hips to foot
    leg_chain = ['Hips', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase', 'LeftToeBase_EndSite']
    leg_y = 0.0
    for joint in leg_chain:
        if joint in offsets:
            leg_y += offsets[joint][1]
    
    return head_y, leg_y, head_chain, leg_chain

def main():
    bvh_file = '/home/davide/PunchRecognition/Data/Session2/Optitrack - 60Hz/Davide 1 - Take 2026-01-14 04.29.52 PM_004_Skeleton 001.bvh'
    
    offsets = parse_bvh_simple(bvh_file)
    
    print("=" * 80)
    print("BVH HEIGHT CALCULATION - DAVIDE SUBJECT")
    print("=" * 80)
    
    # Display all parsed offsets
    print("\nAll joint offsets in BVH file:\n")
    print(f"{'Joint Name':<30} {'X Offset':<15} {'Y Offset':<15} {'Z Offset':<15}")
    print("-" * 75)
    for joint in sorted(offsets.keys()):
        x, y, z = offsets[joint]
        print(f"{joint:<30} {x:>13.6f}  {y:>13.6f}  {z:>13.6f}")
    
    head_y, leg_y, head_chain, leg_chain = calculate_chain_height(offsets)
    
    print("\n" + "=" * 80)
    print("HEAD CHAIN ANALYSIS (Hips → Head → End Site):")
    print("-" * 80)
    cumulative_y = 0
    for joint in head_chain:
        if joint in offsets:
            y_offset = offsets[joint][1]
            cumulative_y += y_offset
            print(f"  {joint:<30} Y = {y_offset:>10.6f}  |  Cumulative = {cumulative_y:>10.6f} cm")
        else:
            print(f"  {joint:<30} (NOT FOUND)")
    
    print(f"\nTop of skeleton (Head End Site): Y = {head_y:.6f} cm\n")
    
    print("=" * 80)
    print("LEFT LEG CHAIN ANALYSIS (Hips → Foot → End Site):")
    print("-" * 80)
    cumulative_y = 0
    for joint in leg_chain:
        if joint in offsets:
            y_offset = offsets[joint][1]
            cumulative_y += y_offset
            print(f"  {joint:<30} Y = {y_offset:>10.6f}  |  Cumulative = {cumulative_y:>10.6f} cm")
        else:
            print(f"  {joint:<30} (NOT FOUND)")
    
    print(f"\nBottom of skeleton (Toe Base End Site): Y = {leg_y:.6f} cm\n")
    
    # Calculate height
    height_cm = head_y - leg_y
    height_inches = height_cm / 2.54
    height_feet = height_cm / 30.48
    
    feet = int(height_feet)
    remaining_inches = (height_feet - feet) * 12
    
    print("=" * 80)
    print("SUBJECT HEIGHT CALCULATION:")
    print("=" * 80)
    print(f"  Height = Top Y - Bottom Y")
    print(f"  Height = {head_y:.6f} cm - ({leg_y:.6f} cm)")
    print(f"  Height = {height_cm:.2f} cm")
    print(f"\n  ➜ {height_inches:.2f} inches")
    print(f"  ➜ {height_feet:.2f} feet")
    print(f"  ➜ {feet}' {remaining_inches:.1f}\"")
    print("=" * 80)

if __name__ == '__main__':
    main()
