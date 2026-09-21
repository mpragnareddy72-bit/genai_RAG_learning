function ChatMessage({ message }) {
    return (
        <div className={`message ${message.role}`}>
            <div className="message-label">
                {message.role === "user" ? "You" : "AI"}
            </div>

            <div className="message-content">
                {message.content}
            </div>
        </div>
    );
}

export default ChatMessage;