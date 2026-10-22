import ollama as ol
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report


def main():
    #Load the dataset
    input_file = "LabeledFake-1K.csv"  # Replace with your dataset file path
    output_file = "annotated_news1.csv"  # Output file for annotated results
    df = pd.read_csv(input_file)

    # Use only the first 10 rows
    #df = df.head(10)  # Take the first 10 rows of the dataset

    # Ground truth labels
    ground_truth = df['label'].tolist()
    predictions = []

    # Define the initial instructions for the      model
    messages = [
        {
            'role': 'system',
            'content': """You are a Bangla fake news detection expert.Answer deterministically and provide only the required label.  Your task is to analyze the given news article step by step. Consider the following questions:

        1. **Headline Analysis**: Is the headline exaggerated or weird to hear then should be fake
        2. **Content Verification**: Does the content make any sense or does it sound logical? content should be similar to real logical events
        3) You must provide just a single sentence of reasoning to support your decision. It must be structured in the following manner:
"Reason: [Your single reason statement]"
        VERY IMPORTANT: Please provide just one sentence of reasoning to support your decision.
        After answering these questions, label whether the news is:
        - 1 if the article is factual and credible.
        - 0 if the article is sensational or illogical to hear .

        For each article news, respond with only the corresponding label (0, 1).
        - Label: [1/0]
        - Reason: [Provide a reason based on the analysis above in one sentence,not more than 1 sentence]

        Example:
        Label: 0
        Reason: The headline is illogical or satirical.""",
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
            content = row['content']

            # Create the input message for the model
            user_message = f"""{content[:1000]}  # Only the first 1000 characters if content is too long
"""
            # Append the user input to the conversation
            messages.append({'role': 'user', 'content': user_message})

            # Build the messages to send, keeping the last N messages within the max history length
            messages_to_send = [messages[0]] + messages[-(max_history_length - 1):]

            # Send the conversation to the model
            try:
                response = ol.chat(model='tmp3myllama3.2', messages=messages_to_send)

                # Get the model's response
                output = response['message']['content'].strip()

                # Append the assistant's response to the conversation
                messages.append({'role': 'assistant', 'content': output})



                Label = output.split("Label:")[1].split("Reason:")[0].strip()
                reason = output.split("Reason:")[1].strip()
                    # Append the decision to predictions list
                #predictions.append(Label)

                # Write the results to the output CSV file

                output_csv.write(f'"{article_id}","{Label}"\n')

                # Optional: Print progress
                #print(f"Annotated Article ID: {article_id} - Label: {Label} ")


            except Exception as e:
                print(f"Error processing Article ID: {article_id} - {e}")
                #predictions.append(-1)  # Append -1 for errors
                output_csv.write(f'"{article_id}",-1\n')
    print('annotation done')
        # Read CSV files into dataframes
    llm_csv_path = "annotated_news1.csv"
    human_csv_path = "LabeledFake-1K.csv"
    llm_df = pd.read_csv(llm_csv_path)
    human_df = pd.read_csv(human_csv_path)
    #human_df=human_df.head(10)

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

    #save to result file
    output_path = "metrics_analysis_banfak_test_COT.txt"
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
