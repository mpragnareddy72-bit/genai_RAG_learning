const API_URL = "http://127.0.0.1:8000";

export async function askQuestion(question, topK = 3) {
    const response = await fetch(`${API_URL}/rag/ask`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({
            question: question,
            top_k: topK,
        }),
    });

    if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));

        throw new Error(
            errorData.detail || "Failed to get answer from backend"
        );
    }

    return response.json();
}