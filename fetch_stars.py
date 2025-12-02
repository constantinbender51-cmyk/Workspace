import pandas as pd
import gdown
import os

# Google Drive file ID extracted from the URL
file_id = '1pBl_UvRgrCuk6cV06y7-R6gRNB-mr0in'
url = f'https://drive.google.com/uc?id={file_id}'
output_file = 'exoplanets.csv'

# Download the file using gdown
print(f'Downloading {output_file} from Google Drive...')
gdown.download(url, output_file, quiet=False)

# Check if file was downloaded successfully
if os.path.exists(output_file):
    print(f'Successfully downloaded {output_file}')
    
    # Load the CSV file
    try:
        # First try reading with comment handling to skip the first 96 comment lines
        # Common comment characters in CSV files: '#', ';', '//', '!'
        df = pd.read_csv(output_file, comment='#')
        print(f'Loaded CSV with {len(df)} rows and {len(df.columns)} columns')
        
        # If still getting parsing errors, try with other comment characters
        if len(df) == 0 or len(df.columns) == 0:
            print('Trying with different comment characters...')
            for comment_char in [';', '!', '//', '%', '*']:
                try:
                    df = pd.read_csv(output_file, comment=comment_char)
                    if len(df) > 0 and len(df.columns) > 0:
                        print(f'Successfully loaded with comment={comment_char}: {len(df)} rows, {len(df.columns)} columns')
                        break
                except:
                    continue
        
        # If still problematic, try skipping rows
        if len(df) == 0 or len(df.columns) == 0:
            print('Trying to skip first 96 rows...')
            df = pd.read_csv(output_file, skiprows=96)
            print(f'Loaded CSV by skipping 96 rows: {len(df)} rows and {len(df.columns)} columns')
        
        # Final fallback with flexible parsing
        if len(df) == 0 or len(df.columns) == 0:
            print('Using flexible parsing with python engine...')
            df = pd.read_csv(output_file, engine='python', on_bad_lines='skip')
            print(f'Loaded CSV with flexible parsing: {len(df)} rows and {len(df.columns)} columns')
        
        # Check if 'sy_dist' column exists
        if 'sy_dist' in df.columns:
            # Sort by distance (ascending) to get closest stars
            df_sorted = df.sort_values(by='sy_dist')
            
            # Get the 5 closest stars
            closest_stars = df_sorted.head(5)
            
            print('\n5 Closest Stars (based on sy_dist):')
            print('=' * 50)
            
            # Display relevant information
            display_columns = ['sy_dist']
            # Add other potentially interesting columns if they exist
            for col in ['pl_name', 'hostname', 'sy_pnum', 'discoverymethod']:
                if col in df.columns:
                    display_columns.append(col)
            
            # Display the closest stars
            print(closest_stars[display_columns].to_string(index=False))
            
            # Save results to a file
            closest_stars.to_csv('closest_stars.csv', index=False)
            print(f'\nResults saved to closest_stars.csv')
        else:
            print("Error: 'sy_dist' column not found in the CSV file.")
            print(f"Available columns: {list(df.columns)}")
            
    except pd.errors.ParserError as e:
        print(f'Parser error: {e}')
        print('Trying with error handling and different parameters...')
        
        # Try reading with error handling for bad lines
        df = pd.read_csv(output_file, on_bad_lines='skip')
        print(f'Loaded CSV with error handling: {len(df)} rows and {len(df.columns)} columns')
        
        # Check if 'sy_dist' column exists
        if 'sy_dist' in df.columns:
            # Sort by distance (ascending) to get closest stars
            df_sorted = df.sort_values(by='sy_dist')
            
            # Get the 5 closest stars
            closest_stars = df_sorted.head(5)
            
            print('\n5 Closest Stars (based on sy_dist):')
            print('=' * 50)
            
            # Display relevant information
            display_columns = ['sy_dist']
            # Add other potentially interesting columns if they exist
            for col in ['pl_name', 'hostname', 'sy_pnum', 'discoverymethod']:
                if col in df.columns:
                    display_columns.append(col)
            
            # Display the closest stars
            print(closest_stars[display_columns].to_string(index=False))
            
            # Save results to a file
            closest_stars.to_csv('closest_stars.csv', index=False)
            print(f'\nResults saved to closest_stars.csv')
        else:
            print("Error: 'sy_dist' column not found in the CSV file.")
            print(f"Available columns: {list(df.columns)}")
            
    except Exception as e:
        print(f'Error reading CSV file: {e}')
else:
    print(f'Error: Failed to download {output_file}')