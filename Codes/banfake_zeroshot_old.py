"""
Part 1 (Python scripts) - see README.md, "Part 1: RevBangla and BanFakeNews".

Dataset   : BanFakeNews (fake news detection: 0 = fake, 1 = authentic)
Prompting : Zero-shot (headline)
Model     : tmp3myllama3.2 (llama3.2:3b, temperature 0.25)
Input     : ../Datasets/BanFakeNews.csv
Outputs   : annotated_news1_zeroshot.csv
            metrics_analysis_banfak_test_zeroshot.txt
Prompt    : Dataset prompting/Banfake/

How to run (from the Codes/ folder, with Ollama running):
    python banfake_zeroshot_old.py
"""
import ollama as ol
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report


def main():
    # Load the dataset
    input_file = "../Datasets/BanFakeNews.csv"  # Replace with your dataset file path
    output_file = "annotated_news1_zeroshot.csv"  # Output file for annotated results
    df = pd.read_csv(input_file)

    # Use only the first 10 rows
    #df = df.head(10)  # Take the first 10 rows of the dataset

    predictions = []

    # Define the initial instructions for the      model
    messages = [
        {
            'role': 'system',
            'content': """You are a Bangla fake news detection expert. Classify the following news articles into one of the two categories:

            - 1: Authentic news.
            - 0: fake news .

        For each headlines of the article news, respond with only the corresponding label (0, 1).Do not provide any explanations.

        headline: "ফণীর পর এবার ধেয়ে আসছে ঘুর্ণিঝড় "GPA - 5", সারাদেশে ৯ নম্বর বিপদ সংকেত জারি! - Bengal Beats"
        Label:
        """,

        }

    ]

    # Set the maximum number of messages to include to stay within the model's context window
    max_history_length = 1500  # Adjust based on the model's capabilities

    # Open the output file to save results
    with open(output_file, "w") as output_csv:
        output_csv.write("headline,Label\n")  # Write CSV header

        # Iterate over each row in the dataset (limited to 10 rows)
        for line in df["headline"]:
            # Extract relevant data for annotation
            #article_id = row['articleID']
            # domain = row['domain']
            # date = row['date']
            # category = row['category']
            # headline = row['headline']
            # content = row['content']

            line_csv = line.replace('"', '""')  # escape quotes for CSV output

            # Create the input message for the model

            # Append the user input to the conversation
            messages.append({'role': 'user', 'content': f'Input: {line}'})

            # Build the messages to send, keeping the last N messages within the max history length
            messages_to_send = [messages[0]] + messages[-(max_history_length - 1):]

            # Send the conversation to the model
            try:
                response = ol.chat(model='tmp3myllama3.2', messages=messages_to_send)

                # Get the model's response
                output = response['message']['content'].strip()

                # Append the assistant's response to the conversation
                messages.append({'role': 'assistant', 'content': output})

                # Extract the bucket ID number from the model's response

                label = output.replace("Label: ", "").strip()
                if label not in ["0", "1"]:
                    label = "-1"
                # Append the decision to predictions list
                #predictions.append(Label)

                # Write the results to the output CSV file

                output_csv.write(f'"{line_csv}","{label}"\n')

                # Optional: Print progress

                print(f">>> {line}\n{label}\n")

            except Exception as e:
                print(f"Error processing headline: {line} - {e}")
                output_csv.write(f'"{line_csv}",-1\n')
    print('annotation done')
        # Read CSV files into dataframes
    llm_csv_path = "annotated_news1_zeroshot.csv"
    human_csv_path = "../Datasets/BanFakeNews.csv"
    llm_df = pd.read_csv(llm_csv_path)
    human_df = pd.read_csv(human_csv_path)
    #human_df=human_df.head(10)

    # Extract labels
    llm_labels = llm_df['Label']
    human_labels = human_df['Label']

    # Calculate accuracy
    # Calculate metrics
    accuracy = accuracy_score(human_labels, llm_labels)
    precision = precision_score(human_labels, llm_labels, average='weighted')
    recall = recall_score(human_labels, llm_labels, average='weighted')
    f1 = f1_score(human_labels, llm_labels, average='weighted')
    conf_matrix = confusion_matrix(human_labels, llm_labels)
    report = classification_report(human_labels, llm_labels)
    print(f"\nAccuracy of LLM-generated annotations: {accuracy * 100:.2f}%")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(report)

    #save to result file
    output_path = "metrics_analysis_banfak_test_zeroshot.txt"
    with open(output_path, "w") as result_file:
        result_file.write(f"Metrics Analysis:\n")
        result_file.write(f"Accuracy: {accuracy:.2f}\n")
        result_file.write(f"Precision: {precision:.2f}\n")
        result_file.write(f"Recall: {recall:.2f}\n")
        result_file.write(f"F1 Score: {f1:.2f}\n")
        result_file.write("\nConfusion Matrix:\n")
        result_file.write(f"{conf_matrix}\n")
        result_file.write("\nClassification Report:\n")
        result_file.write(f"{report}\n")
if __name__ == "__main__":
    main()
