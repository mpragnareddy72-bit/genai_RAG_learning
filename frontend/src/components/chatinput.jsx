import { useState } from "react";

function ChatInput({ onSend, loading }) {
    const [question, setQuestion] = useState("");

    const handleSubmit = async (event) => {
        event.preventDefault();

        const trimmedQuestion = question.trim();

        if (!trimmedQuestion || loading) {
            return;
        }

        await onSend(trimmedQuestion);

        setQuestion("");
    };

    return (
        <form className="chat-input" onSubmit={handleSubmit}>
            <input
                type="text"
                value={question}
                onChange={(event) => setQuestion(event.target.value)}
                placeholder="Ask something about your documents..."
                disabled={loading}
            />

            <button type="submit" disabled={loading || !question.trim()}>
                {loading ? "Thinking..." : "Send"}
            </button>
        </form>
    );
}

export default ChatInput;