import ollama as ol
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report


def main():
    # Load the dataset
    input_file = "data.csv"  # Replace with your dataset file path
    output_file = "annotated_data_fewshot_1B.csv"  # Output file for annotated results
    df = pd.read_csv(input_file)

    # Use only the first 10 rows
    #df = df.head(400)  # Take the first 10 rows of the dataset

    # Define the initial instructions for the model
    messages = [
        {
            'role': 'system',
            'content': """You are an expert in  sentiment analysis in Bangla. You must Classify the sentiment of the following product reviews into one of the two categories only, not any other explanation :

            - positive: Positive sentiment
            - negative: Negative sentiment

        For each product review comment, respond with only the corresponding label ("positive" or "negative").
        **Important** : Do not provide any other explanations or additional text. Only respond with positive or negative
        Example Inputs:
        comment: "চমৎকার একটি প্রোডাক্ট"
        sentiment:positive
        
        comment: "বেল্ট আরও মোটা হওয়া উচিত ছিল কোয়ালিটি মোটামুটি দাম কমালে ভালো হবে"
        sentiment:negative
        
        comment: "খুব বাজে দামটাও অনেক বেশি"
        sentiment:negative
        
        
        """,

        }

    ]

    # Set the maximum number of messages to include to stay within the model's context window
    max_history_length = 1500  # Adjust based on the model's capabilities

    # Open the output file to save results
    with open(output_file, "w") as output_csv:
        output_csv.write("comment,sentiment\n")  # Write CSV header

        # Iterate over each row in the dataset (limited to 10 rows)
        for _, row in df.iterrows():
            # Extract relevant data for annotation
            comment = row['comment']
            # Create the input message for the model

            # Append the user input to the conversation
            messages.append({'role': 'user', 'content': f'Input: {comment}'})

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

                sentiment = output.replace("sentiment: ", "").strip()
                #print(type(label))
                # Write the results to the output CSV file
                if sentiment not in ["positive", "negative"]:
                    sentiment = "unknown"

                output_csv.write(f'"{comment}","{sentiment}"\n')

                # Optional: Print progress

                print(f">>> {comment}\n{sentiment}\n")

            except Exception as e:
                print(f"Error processing Article ID: {comment} - {e}")

                output_csv.write(f'"{comment}",-1\n')
    print('annotation done')
        # Read CSV files into dataframes
    llm_csv_path = "annotated_data_fewshot_1B.csv"
    human_csv_path = "data.csv"
    llm_df = pd.read_csv(llm_csv_path)
    human_df = pd.read_csv(human_csv_path)
    #human_df=human_df.head(400)

    # Extract labels
    llm_labels = llm_df['sentiment']
    human_labels = human_df['sentiment']
    print('predicted value',llm_labels.unique())
    print('human label',human_labels.unique())

    # Handle NaN values in predictions
    llm_labels = llm_labels.fillna("unknown")  # Replace NaN with "unknown"

    # Filter valid predictions (only "negative" or "positive")
    valid_indices = llm_labels.isin(["negative", "positive"])
    filtered_llm_labels = llm_labels[valid_indices]
    filtered_human_labels = human_labels[valid_indices]

    # Calculate accuracy
    accuracy = accuracy_score(filtered_human_labels, filtered_llm_labels)

    # Calculate accuracy
    # Calculate metrics
    #accuracy = accuracy_score(human_labels, llm_labels)
    precision = precision_score(filtered_human_labels, filtered_llm_labels, average='weighted')
    recall = recall_score(filtered_human_labels, filtered_llm_labels, average='weighted')
    f1 = f1_score(filtered_human_labels, filtered_llm_labels, average='weighted')
    conf_matrix = confusion_matrix(filtered_human_labels, filtered_llm_labels)
    report = classification_report(filtered_human_labels, filtered_llm_labels)
    print(f"\nAccuracy of LLM-generated annotations: {accuracy * 100:.2f}%")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(report)
    #
    # #save to result file
    output_path = "metrics_analysis_Revbangla_test_fewshot_1B.txt"
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
