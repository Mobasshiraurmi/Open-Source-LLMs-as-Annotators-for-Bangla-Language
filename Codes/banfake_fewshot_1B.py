import ollama as ol
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    classification_report


def main():
    # Load the dataset
    input_file = "LabeledFake-1K.csv"  # Replace with your dataset file path
    output_file = "annotated_news1_fewshot_1B.csv"  # Output file for annotated results
    df = pd.read_csv(input_file)

    # Use only the first 10 rows
    # df = df.head(10)  # Take the first 10 rows of the dataset

    predictions = []

    # Define the initial instructions for the      model
    messages = [
        {
            'role': 'system',
            'content': """You are a Bangla fake news detection expert. Classify the following news articles into one of the two categories:

            - 1: Authentic news.
            - 0: fake news .

        For each headlines of the article news, respond with only the corresponding label (0, 1).Do not provide any explanations.
        Example Inputs:
        headline: "ফণীর পর এবার ধেয়ে আসছে ঘুর্ণিঝড় "GPA - 5", সারাদেশে ৯ নম্বর বিপদ সংকেত জারি! - Bengal Beats"
        Label:0

        headline: "মুরগির হামলায় শেয়াল নিহত"
        Label:0

        headline: "বিদেশ থেকে উন্নতমানের বিরোধীদল আমদানি করার পরামর্শ দিলেন অনলাইন রাজনীতি বিশেষজ্ঞরা"
        Label:0



        """,

        }

    ]

    # Set the maximum number of messages to include to stay within the model's context window
    max_history_length = 1500  # Adjust based on the model's capabilities

    # Open the output file to save results
    with open(output_file, "w") as output_csv:
        output_csv.write("articleID,Label\n")  # Write CSV header

        # Iterate over each row in the dataset (limited to 10 rows)
        for _, row in df.iterrows():
            # Extract relevant data for annotation
            article_id = row['articleID']
            # domain = row['domain']
            # date = row['date']
            # category = row['category']
            headline = row['headline']
            # content = row['content']

            # Create the input message for the model

            # Append the user input to the conversation
            messages.append({'role': 'user', 'content': f'Input: {headline}'})

            # Build the messages to send, keeping the last N messages within the max history length
            messages_to_send = [messages[0]] + messages[-(max_history_length - 1):]

            # Send the conversation to the model
            try:
                response = ol.chat(model='1bmyllama3.2', messages=messages_to_send)

                # Get the model's response
                output = response['message']['content'].strip()

                # Append the assistant's response to the conversation
                messages.append({'role': 'assistant', 'content': output})

                # Extract the bucket ID number from the model's response

                label = output.replace("Label: ", "").strip()
                # Write the results to the output CSV file
                if label not in ["0", "1"]:
                    label = "-1"
                output_csv.write(f'"{article_id}","{label}"\n')

                # Optional: Print progress

                print(f">>> {headline}\n{label}\n")

            except Exception as e:
                print(f"Error processing Article ID: {article_id} - {e}")
                predictions.append(-1)  # Append -1 for errors
                output_csv.write(f'"{article_id}",-1\n')
    print('annotation done')
    # Read CSV files into dataframes
    llm_csv_path = "annotated_news1_fewshot_1B.csv"
    human_csv_path = "LabeledFake-1K.csv"
    llm_df = pd.read_csv(llm_csv_path)
    human_df = pd.read_csv(human_csv_path)
    # human_df=human_df.head(10)

    # Extract labels
    llm_labels = llm_df['Label']
    human_labels = human_df['label']

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

    # save to result file
    output_path = "metrics_analysis_banfak_test_fewshot_1B.txt"
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
