import { useState } from "react";
import ChatMessage from "./components/ChatMessage";
import ChatInput from "./components/ChatInput";
import { askQuestion } from "./services/api";
import "./App.css";

function App() {
    const [messages, setMessages] = useState([]);
    const [loading, setLoading] = useState(false);

    const handleSend = async (question) => {
        const userMessage = {
            role: "user",
            content: question,
        };

        setMessages((previous) => [
            ...previous,
            userMessage,
        ]);

        setLoading(true);

        try {
            const data = await askQuestion(question, 3);

            const assistantMessage = {
                role: "assistant",
                content: data.answer,
            };

            setMessages((previous) => [
                ...previous,
                assistantMessage,
            ]);
        } catch (error) {
            const errorMessage = {
                role: "assistant",
                content: `Error: ${error.message}`,
            };

            setMessages((previous) => [
                ...previous,
                errorMessage,
            ]);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="app">
            <header className="header">
                <h1>RAG Assistant</h1>
                <p>
                    Nomic + BM25 + RRF + Reranker + Ollama
                </p>
            </header>

            <main className="chat-container">
                {messages.length === 0 ? (
                    <div className="welcome">
                        <h2>Ask your documents</h2>

                        <p>
                            Ask a question and the RAG system will
                            retrieve relevant information and generate
                            an answer.
                        </p>

                        <div className="example">
                            Example:
                            <br />
                            <strong>
                                What is the population of Afghanistan?
                            </strong>
                        </div>
                    </div>
                ) : (
                    <div className="messages">
                        {messages.map((message, index) => (
                            <ChatMessage
                                key={index}
                                message={message}
                            />
                        ))}

                        {loading && (
                            <div className="message assistant">
                                <div className="message-label">
                                    AI
                                </div>

                                <div className="message-content">
                                    Thinking...
                                </div>
                            </div>
                        )}
                    </div>
                )}
            </main>

            <ChatInput
                onSend={handleSend}
                loading={loading}
            />
        </div>
    );
}

export default App;