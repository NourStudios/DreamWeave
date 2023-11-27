def calculate_similarity(predicted_values, trained_values, threshold=0.1):
    matched_pixels = 0
    total_pixels = len(predicted_values)

    for predicted_value, trained_value in zip(predicted_values, trained_values):
        if abs(trained_value - predicted_value) <= threshold:
            matched_pixels += 1

    similarity = (matched_pixels / total_pixels) * 100
    return similarity

# Example usage
predicted_values = [1.00, 0.00, 1.00, 0.00, 2.00, 2.00, 2.00]
trained_values = [1.42, 1.41, 1.39, 1.36, 1.86, 1.86, 1.86]

similarity = calculate_similarity(predicted_values, trained_values)
print(f"Similarity: {similarity:.2f}%")
