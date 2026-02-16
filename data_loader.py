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
    Parses B3FD images by filename pattern: ID_BirthDate_YearTaken.jpg
    Example: 41221068_1946-02-26_1968.jpg
    """
    print(f"   Searching for B3FD images taken after {min_year}...")
    data = []
    
    # Recursively walk through EVERY folder in datasets
    for root, dirs, files in os.walk(dataset_root):
        for file in files:
            # Optimization: Only process .jpg files
            if not file.lower().endswith('.jpg'):
                continue
                
            # Parse Pattern: ID_BirthDate_YearTaken.jpg
            # We split by '_'
            parts = Path(file).stem.split('_')
            
            # Must have at least 3 parts (ID, Date, Year)
            if len(parts) >= 3:
                try:
                    # 1. Extract Year Taken (Last part)
                    year_taken_str = parts[-1]
                    # Handle edge cases like "1980(1).jpg"
                    if '(' in year_taken_str: year_taken_str = year_taken_str.split('(')[0]
                    year_taken = int(year_taken_str)
                    
                    # 2. Filter by Year
                    if year_taken < min_year:
                        continue
                        
                    # 3. Extract Birth Year (Second part: 1946-02-26)
                    birth_date = parts[1]
                    if '-' not in birth_date: continue
                    birth_year = int(birth_date.split('-')[0])
                    
                    # 4. Calculate Age
                    age = year_taken - birth_year
                    
                    # 5. Sanity Check
                    if age < 0 or age > 116:
                        continue
                    
                    # 6. Add to list (Gender is -1 as requested)
                    data.append({
                        'image_path': Path(root) / file,
                        'age': age,
                        'gender': -1,
                        'source': 'b3fd'
                    })
                    
                except (ValueError, IndexError):
                    # Skip files that don't match the math/format
                    continue

    print(f"   Loaded {len(data)} B3FD images.")
    return data

def get_unified_dataset(dataset_dir='./datasets'):
    """Main function to call from your notebook"""
    root = Path(dataset_dir)
    print("Scanning datasets...")
    
    data = []
    data.extend(load_utkface(root))
    data.extend(load_adience(root))
    data.extend(load_fgnet(root))
    
    # Load B3FD (Defaults to 1980+)
    data.extend(load_b3fd(root, min_year=1980))
    
    df = pd.DataFrame(data)
    print(f"Loaded {len(df)} images total.")
    if not df.empty:
        print(df['source'].value_counts())
    return df