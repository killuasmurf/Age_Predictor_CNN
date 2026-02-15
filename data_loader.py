import pandas as pd
import os
import re
from pathlib import Path

def load_utkface(dataset_root):
    """Parses UTKFace"""
    data = []
    # Try multiple common paths
    paths = [
        dataset_root / "UTKFace",
        dataset_root / "UTKFace" / "UTKFace"
    ]
    path = next((p for p in paths if p.exists()), None)
    
    if not path:
        return []
    
    for file in os.listdir(path):
        split = file.split('_')
        if len(split) == 4:
            try:
                age = int(split[0])
                gender = int(split[1]) # 0=Male, 1=Female
                data.append({'image_path': path / file, 'age': age, 'gender': gender, 'source': 'utkface'})
            except: continue
    return data

def load_adience(dataset_root):
    """Parses Adience (Robust Search)"""
    data = []
    print("   Searching for Adience files...")
    fold_files = list(dataset_root.rglob("fold_*_data.txt"))
    
    if not fold_files:
        return []
    
    adience_base_dir = fold_files[0].parent
    
    def parse_age(s):
        try: return int((int(s[1:].split(',')[0]) + int(s[:-1].split(', ')[1])) / 2)
        except: return None

    for txt_file in fold_files:
        with open(txt_file, 'r') as f:
            next(f) # Skip header
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 5:
                    user_id, filename, face_id, age_str, gender_str = parts[:5]
                    age = parse_age(age_str)
                    gender = 0 if gender_str == 'm' else 1 if gender_str == 'f' else None
                    
                    img_names = [
                        f"coarse_tilt_aligned_face.{face_id}.{filename}",
                        f"landmark_aligned_face.{face_id}.{filename}"
                    ]
                    
                    found_path = None
                    for name in img_names:
                        p1 = adience_base_dir / "faces" / user_id / name
                        p2 = adience_base_dir / "aligned" / user_id / name
                        if p1.exists(): found_path = p1
                        elif p2.exists(): found_path = p2
                        if found_path: break
                    
                    if age is not None and gender is not None and found_path:
                        data.append({'image_path': found_path, 'age': age, 'gender': gender, 'source': 'adience'})
    return data

def load_fgnet(dataset_root):
    """Parses FGNET"""
    data = []
    path = dataset_root / "FGNET/images" 
    if not path.exists(): path = dataset_root / "FGNET"
    if not path.exists(): return []

    pattern = re.compile(r"(\d+)A(\d+)") 
    for file in os.listdir(path):
        match = pattern.search(file)
        if match:
            age = int(match.group(2))
            data.append({'image_path': path / file, 'age': age, 'gender': -1, 'source': 'fgnet'})
    return data

def load_b3fd(dataset_root, min_year=1980):
    """
    Parses B3FD based on folder structure: 
    root/B3FD/PersonName/ID_BirthDate_YearTaken.jpg
    Example: 41221068_1946-02-26_1968.jpg
    """
    print(f"   Searching for B3FD images taken after {min_year}...")
    data = []
    
    # 1. Locate the B3FD root folder
    # It might be nested, so we search for a folder literally named "B3FD"
    b3fd_roots = list(dataset_root.rglob("B3FD"))
    if not b3fd_roots:
        print("   ⚠️ B3FD folder not found!")
        return []
    
    # Use the first valid B3FD folder found
    dataset_path = b3fd_roots[0]
    
    # 2. Walk through all person folders
    # The structure is dataset_path / PersonName / Image.jpg
    for person_folder in dataset_path.iterdir():
        if not person_folder.is_dir():
            continue
            
        for img_file in person_folder.glob("*.jpg"):
            # Parse Filename: 41221068_1946-02-26_1968.jpg
            # Parts: [ID, BirthDate, YearTaken]
            parts = img_file.stem.split('_')
            
            if len(parts) >= 3:
                try:
                    # Extract Year Taken (last part)
                    year_taken = int(parts[-1])
                    
                    # FILTER: Only want images after 1980
                    if year_taken < min_year:
                        continue
                        
                    # Extract Birth Year to calculate Age
                    birth_date = parts[1] # "1946-02-26"
                    birth_year = int(birth_date.split('-')[0])
                    
                    age = year_taken - birth_year
                    
                    # Sanity check for age (0 to 116)
                    if age < 0 or age > 116:
                        continue

                    # Gender is NOT in the filename.
                    # We can try to infer it if you have a metadata CSV, 
                    # but for now, we set it to -1 (unknown) or skip if you strictly need gender.
                    # If you strictly need gender, we MUST use the CSV method or an external lookup.
                    # HOWEVER, usually B3FD comes with a CSV mapping "PersonName" -> "Gender".
                    # Let's check for a metadata file nearby to build a gender map.
                    
                    # For now, to get the code running, we will set Gender to -1.
                    # You can use masked loss to handle this, OR we can try to find the CSV.
                    
                    data.append({
                        'image_path': img_file,
                        'age': age,
                        'gender': -1, # Placeholder (see note below)
                        'source': 'b3fd'
                    })
                    
                except ValueError:
                    continue

    # --- OPTIONAL: Try to fix Gender from CSV ---
    # If we collected data, let's try to find a CSV to fill in the genders
    if data:
        print(f"   Found {len(data)} images. Attempting to load gender metadata...")
        csv_files = list(dataset_root.rglob("*.csv"))
        gender_map = {}
        
        for csv in csv_files:
            try:
                df = pd.read_csv(csv)
                # Look for columns like 'name'/'person' and 'gender'
                df.columns = [c.lower() for c in df.columns]
                if 'name' in df.columns and 'gender' in df.columns:
                    # Create a map: "PersonName" -> 0/1
                    for _, row in df.iterrows():
                        g = row['gender']
                        if isinstance(g, str):
                            g = 0 if g.lower() in ['male', 'm'] else 1
                        gender_map[row['name']] = g
                    break # Stop after finding the first valid metadata file
            except: continue
        
        if gender_map:
            print(f"   Loaded gender labels for {len(gender_map)} people.")
            # Update the data list with genders
            valid_data = []
            for item in data:
                # Person folder name is the parent of the image
                person_name = item['image_path'].parent.name
                if person_name in gender_map:
                    item['gender'] = gender_map[person_name]
                    valid_data.append(item)
            data = valid_data
        else:
            print("   ⚠️ No gender metadata found. Returning data with Gender = -1.")

    print(f"   Loaded {len(data)} B3FD images (Post-1980).")
    return data
def get_unified_dataset(dataset_dir='./datasets'):
    """Main function to call from your notebook"""
    root = Path(dataset_dir)
    print("Scanning datasets...")
    
    data = []
    data.extend(load_utkface(root))
    data.extend(load_adience(root))
    data.extend(load_fgnet(root))
    data.extend(load_b3fd(root))
    
    df = pd.DataFrame(data)
    print(f"Loaded {len(df)} images total.")
    print(df['source'].value_counts())
    return df