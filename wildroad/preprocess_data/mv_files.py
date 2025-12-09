import os
import shutil
import glob
from pathlib import Path

def move_tile_folders(source_dir, output_dir):
    """
    move all tile_{id} folders from GLG_* folders to the output directory

    Args:
        source_dir: source directory path (contains GLG_0, GLG_1 etc. folders)
        output_dir: output directory path
    """
    
    # ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    # statistics
    total_moved = 0
    total_files = 0
    errors = []
    
    print(f"🔍 scan source directory: {source_path}")
    print(f"📁 output directory: {output_path}")
    print("-" * 60)
    
    # find all GLG_* folders
    glg_folders = sorted([f for f in source_path.iterdir() 
                         if f.is_dir() and f.name.startswith('GLG_')])
    
    if not glg_folders:
        print("⚠️  no GLG_* folders found")
        return
    
    print(f"📋 found {len(glg_folders)} GLG folders")
    print()
    
    # iterate over each GLG folder
    for glg_folder in glg_folders:
        print(f"🔄 processing folder: {glg_folder.name}")
        
        # find all tile_* folders in the GLG folder
        tile_pattern = glg_folder / "tile_*"
        tile_folders = glob.glob(str(tile_pattern))
        
        if not tile_folders:
            print(f" ⚠️  no tile_* folders found")
            continue
        
        print(f" 📦 found {len(tile_folders)} tile folders")
        
        # move each tile folder
        for tile_folder in tile_folders:
            tile_path = Path(tile_folder)
            tile_name = tile_path.name
            
            try:
                # destination path
                dest_path = output_path / tile_name
                
                # check if the destination already exists
                if dest_path.exists():
                    print(f" ⚠️  destination already exists: {tile_name} (skip)")
                    continue
                
                # count the number of files in the folder
                file_count = sum(1 for _ in tile_path.rglob('*') if _.is_file())
                
                # move the folder
                shutil.move(str(tile_path), str(dest_path))
                
                total_moved += 1
                total_files += file_count
                
                print(f"   ✅ moved: {tile_name} ({file_count} files)")
                
            except Exception as e:
                error_msg = f"error moving {tile_name}: {str(e)}"
                errors.append(error_msg)
                print(f"   ❌ {error_msg}")
        
        print()
    
    # print summary information
    print("=" * 60)
    print("📊 move completed statistics:")
    print(f"   ✅ successfully moved folders: {total_moved} folders")
    print(f"   📄 total files: {total_files} files")
    
    if errors:
        print(f"   ❌ error count: {len(errors)} errors")
        print("\nerror details:")
        for error in errors:
            print(f"   - {error}")
    
    print(f"\n🎉 all tile folders have been moved to: {output_path}")

def verify_move_result(output_dir):
    """
    verify the move result
    """
    output_path = Path(output_dir)
    
    if not output_path.exists():
        print("❌ output directory does not exist")
        return
    
    tile_folders = [f for f in output_path.iterdir() 
                   if f.is_dir() and f.name.startswith('tile_')]
    
    print(f"\n🔍 verify result:")
    print(f"   output directory contains {len(tile_folders)} tile folders")
    
    if tile_folders:
        print("   folder list:")
        for i, folder in enumerate(sorted(tile_folders), 1):
            file_count = sum(1 for _ in folder.rglob('*') if _.is_file())
            print(f"   {i:2d}. {folder.name} ({file_count} files)")

if __name__ == "__main__":
    
    SOURCE_DIR = './wildroad_GLG'
    OUTPUT_DIR = './wildroad_GLG'
    
    # verify the source directory
    if not os.path.exists(SOURCE_DIR):
        print(f"❌ source directory does not exist: {SOURCE_DIR}")
        exit(1)
    
    try:
        # execute the move operation
        move_tile_folders(SOURCE_DIR, OUTPUT_DIR)
        
        # verify the result
        verify_move_result(OUTPUT_DIR)
        
    except KeyboardInterrupt:
        print("\n⚠️  operation interrupted by user")
    except Exception as e:
        print(f"❌ program execution error: {str(e)}")

# if you want to specify the path in the code, you can uncomment the following code:
# SOURCE_DIR = "/path/to/your/source/directory"  # replace with the actual source directory path
# OUTPUT_DIR = "/path/to/your/output/directory"   # replace with the actual output directory path
# move_tile_folders(SOURCE_DIR, OUTPUT_DIR)
# verify_move_result(OUTPUT_DIR)