import os
from pathlib import Path

# Map capture files to likely SM files
raw_data = Path("raw_data")
sm_files = Path("sm_files")

captures = list(raw_data.glob("*.zip"))
print(f"Total captures: {len(captures)}\n")

# Parse capture names and try to match with SM files
matches = []
for cap in sorted(captures):
    cap_name = cap.stem
    # Extract song name (remove difficulty and timestamp)
    parts = cap_name.split('_')
    
    # Common patterns to try
    possible_names = []
    
    # Try various combinations
    if "Lucky_Orb" in cap_name:
        possible_names.append("Lucky Orb.sm")
    elif "Decorator" in cap_name:
        possible_names.append("DECORATOR.sm")
    elif "Charles" in cap_name:
        possible_names.append("Charles.sm")
    elif "Nostalgic" in cap_name:
        possible_names.append("Nostalgic Winds of Autumn.sm")
    elif "39_Music" in cap_name:
        possible_names.append("39 Music!.sm")
    elif "Butterfly_Cat" in cap_name:
        possible_names.append("Butterfly Cat.sm")
    elif "Catch_the_wave" in cap_name or "Catch_The_Wave" in cap_name:
        possible_names.append("Catch The Wave.sm")
    elif "Chigau" in cap_name:
        possible_names.append("Chigau.sm")
    elif "Confession" in cap_name:
        possible_names.append("Confession.sm")
    elif "Even_a_Kunoichi" in cap_name:
        possible_names.append("Even a Kunoichi Needs Love.sm")
    elif "Failure_Girl" in cap_name:
        possible_names.append("Failure Girl.sm")
    elif "Fantasy_Film" in cap_name:
        possible_names.append("Fantasy Film.sm")
    elif "Friend" in cap_name:
        possible_names.append("F(R)IEND.sm")
    elif "Getting_Faster" in cap_name:
        possible_names.append("Getting Faster and Faster.sm")
    elif "Isolation_Thanatos" in cap_name:
        possible_names.append("Isolation=Thanatos.sm")
    elif "Kimagure_Mercy" in cap_name:
        possible_names.append("Kimagure Mercy.sm")
    elif "Little_But_Adult" in cap_name:
        possible_names.append("Little Bit Adult Hit.sm")
    elif "Love_song" in cap_name or "Love_Song" in cap_name:
        possible_names.append("Love Song.sm")
    elif "Melt" in cap_name:
        possible_names.append("Melt.sm")
    elif "Neko_Neko" in cap_name:
        possible_names.append("Neko Neko Super Fever Night.sm")
    elif "Night_Sky" in cap_name:
        possible_names.append("Night Sky Patrol of Tomorrow.sm")
    
    for sm_name in possible_names:
        sm_path = sm_files / sm_name
        if sm_path.exists():
            # Extract difficulty
            diff_str = None
            if "Medium" in cap_name or "Moyen" in cap_name:
                # Extract number after Medium/Moyen
                for i, p in enumerate(parts):
                    if "Medium" in p or "Moyen" in p:
                        if i + 1 < len(parts):
                            try:
                                diff_num = int(parts[i+1].split('-')[0])
                                diff_str = f"medium_{diff_num}"
                            except:
                                pass
                        # Try extracting from same part
                        import re
                        m = re.search(r'(\d+)', p)
                        if m:
                            diff_str = f"medium_{m.group(1)}"
            elif "Easy" in cap_name:
                for i, p in enumerate(parts):
                    if "Easy" in p:
                        if i + 1 < len(parts):
                            try:
                                diff_num = int(parts[i+1].split('-')[0])
                                diff_str = f"easy_{diff_num}"
                            except:
                                pass
                        import re
                        m = re.search(r'(\d+)', p)
                        if m:
                            diff_str = f"easy_{m.group(1)}"
            elif "Hard" in cap_name:
                for i, p in enumerate(parts):
                    if "Hard" in p:
                        if i + 1 < len(parts):
                            try:
                                diff_num = int(parts[i+1].split('-')[0])
                                diff_str = f"hard_{diff_num}"
                            except:
                                pass
                        import re
                        m = re.search(r'(\d+)', p)
                        if m:
                            diff_str = f"hard_{m.group(1)}"
            
            matches.append((cap.name, sm_name, diff_str))
            break

print(f"Matched songs: {len(matches)}\n")
for cap, sm, diff in matches:
    print(f"{cap:60s} -> {sm:40s} ({diff})")
