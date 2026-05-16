import os
import sys
import pandas as pd
import re

# Function to extract the option letter from the text
def extract_option_letter(text):
    if not isinstance(text, str):
        return None
    match = re.search(r'Option:\s*[\[\s]*(\w)', text)
    if match:
        return match.group(1).upper()
    else:
        text_strip = text.strip()
        if text_strip:
            return text_strip[0].upper()
        return None


if __name__ == '__main__':
    # Path to the file containing the model's output results
    # Default to the new agent_run.py output file
    file_path = 'result/gemini-3.1-flash-lite_output4.csv'
    
    # Allow specifying a custom file via command line argument
    if len(sys.argv) > 1:
        file_path = sys.argv[1]

    if not os.path.exists(file_path):
        print(f"Error: Output file not found: {file_path}")
        sys.exit(1)

    # Define the output directory
    output_dir = 'result'
    os.makedirs(output_dir, exist_ok=True)

    # Read the CSV file into a Pandas DataFrame
    df = pd.read_csv(file_path)

    # Adapt to both old and new CSV formats
    if 'extracted_option' in df.columns:
        # Format from agent_run.py
        df['Extracted_Option'] = df['extracted_option'].apply(
            lambda x: str(x).strip().upper() if pd.notna(x) and str(x) != 'ERROR' else None
        )
    elif 'Output' in df.columns:
        # Old format
        df['Extracted_Option'] = df['Output'].apply(extract_option_letter)
    else:
        print("Error: The CSV file does not contain 'extracted_option' or 'Output' columns.")
        sys.exit(1)

    # Drop rows where 'Extracted_Option' is NaN (i.e., no valid option was extracted)
    df_valid = df.dropna(subset=['Extracted_Option'])
    df_valid = df_valid[df_valid['Extracted_Option'] != '']

    # Initialize a list to store accuracy data for each question category
    accuracy_data = []

    if len(df_valid) == 0:
        accuracy_data.append({
            'Category': 'Total',
            'Accuracy': 0.0,
            'Num': 0
        })
    else:
        # Group the DataFrame by the 'question_category' column
        for category, group in df_valid.groupby('question_category'):
            correct_count = (group['Extracted_Option'].astype(str) == group['answer'].astype(str).str.upper()).sum()
            accuracy = correct_count / len(group)
            accuracy_data.append({
                'Category': category,
                'Accuracy': accuracy,
                'Num': len(group)
            })

        # Calculate the overall accuracy across all categories
        total_correct = (df_valid['Extracted_Option'].astype(str) == df_valid['answer'].astype(str).str.upper()).sum()
        total_accuracy = total_correct / len(df_valid)
        accuracy_data.append({
            'Category': 'Total',
            'Accuracy': total_accuracy,
            'Num': len(df_valid)
        })

    # Convert the accuracy data list into a DataFrame
    accuracy_df = pd.DataFrame(accuracy_data)

    # Construct the output file path for the Excel file
    base_name = os.path.basename(file_path)
    output_file_name = base_name.replace('.csv', '.xlsx')
    if 'output' in output_file_name:
        output_file_name = output_file_name.replace('output', 'acc')
    else:
        output_file_name = output_file_name.replace('.xlsx', '_acc.xlsx')
        
    output_excel_path = os.path.join(output_dir, output_file_name)

    # Save the accuracy results to an Excel file
    with pd.ExcelWriter(output_excel_path) as writer:
        accuracy_df.to_excel(writer, index=False, sheet_name='Accuracy Results')

    # Print a message indicating the processing is complete
    print(f"Processed {file_path}")
    print(f"Evaluated {len(df_valid)} valid answers out of {len(df)} total rows.")
    if len(df_valid) > 0:
        print(f"Overall Accuracy: {total_accuracy:.2%}")
    print(f"Saved accuracy results to {output_excel_path}")
