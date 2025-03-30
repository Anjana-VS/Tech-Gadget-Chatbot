import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import ProductDetails from './ProductDetails';
import './App.css';

function Chatbot() {
  const [messages, setMessages] = useState([]);
  const [context, setContext] = useState({});
  const messagesEndRef = useRef(null);
  const hasSentStartMessage = useRef(false);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    if (!hasSentStartMessage.current) {
      hasSentStartMessage.current = true;
      sendMessage("start");
    }
  }, []);

  const sendMessage = async (message) => {
    setMessages((prev) => [...prev, { sender: "user", text: message }]);

    try {
      const response = await axios.post("http://127.0.0.1:8000/chat", {
        message,
        context,
      }, {
        headers: {
          "Content-Type": "application/json",
        },
      });

      const data = response.data;
      setMessages((prev) => [...prev, { sender: "bot", text: data.response }]);
      setContext(data.context);
    } catch (error) {
      console.error("Error sending message:", error);
      setMessages((prev) => [...prev, { sender: "bot", text: "Sorry, there was an error communicating with the server." }]);
    }
  };

  const extractOptions = (text) => {
    const optionsMatch = text.match(/\(options: (.*?)\)/);
    if (optionsMatch) {
      return optionsMatch[1].split(", ");
    }
    return [];
  };

  const handleOptionClick = (option) => {
    sendMessage(option);
  };

  const handleActionClick = (action) => {
    sendMessage(action);
  };

  // Function to render a message, handling the comparison section as a bullet-point list
  const renderMessage = (text) => {
    // Split the message into lines
    const lines = text.split('\n');
    const comparisonStartIndex = lines.findIndex(line => line.includes("Here’s a detailed comparison"));
    if (comparisonStartIndex === -1) {
      // If there's no comparison section, render the text as-is
      return <span>{text}</span>;
    }

    // Extract the comparison section
    const comparisonLines = [];
    let i = comparisonStartIndex + 2; // Skip the "Here’s a detailed comparison..." line and the empty line
    while (i < lines.length && lines[i].startsWith('-')) {
      comparisonLines.push(lines[i].substring(2).trim()); // Remove the "- " prefix
      i++;
    }

    return (
      <div>
        {/* Render the text before the comparison */}
        <span>{lines.slice(0, comparisonStartIndex).join('\n')}</span>
        {/* Render the comparison title */}
        <div className="comparison-title">{lines[comparisonStartIndex]}</div>
        {/* Render the empty line after the title */}
        <span>{lines[comparisonStartIndex + 1]}</span>
        {/* Render the comparison section as a bullet-point list */}
        <ul className="comparison-list">
          {comparisonLines.map((line, index) => (
            <li key={index}>{line}</li>
          ))}
        </ul>
        {/* Render the text after the comparison */}
        <span>{lines.slice(i).join('\n')}</span>
      </div>
    );
  };

  return (
    <div className="chat-container">
      {messages.map((msg, index) => (
        <div key={index} className={`message ${msg.sender}`}>
          {renderMessage(msg.text)}
          {msg.sender === "bot" && extractOptions(msg.text).length > 0 && (
            <div className="options">
              {extractOptions(msg.text).map((option, idx) => (
                <button key={idx} onClick={() => handleOptionClick(option)}>
                  {option}
                </button>
              ))}
            </div>
          )}
          {msg.sender === "bot" && msg.text.includes("proceed with one of these options or stop the process") && (
            <div className="options">
              <button onClick={() => handleActionClick("proceed")}>Proceed</button>
              <button onClick={() => handleActionClick("stop")}>Stop</button>
              <button onClick={() => handleActionClick("explore more")}>Explore More</button>
              {context.recommendation_history?.length > 0 && (
                <button onClick={() => handleActionClick("go back to the previous recommendations")}>
                  Go Back
                </button>
              )}
            </div>
          )}
          {msg.sender === "bot" && msg.text.includes("add more items or finalize your order") && (
            <div className="options">
              <button onClick={() => handleActionClick("explore more")}>Add More Items</button>
              <button onClick={() => handleActionClick("finalize my order")}>Finalize Order</button>
            </div>
          )}
          {msg.sender === "bot" && msg.text.includes("Here are some matching items") && (
            <ProductDetails items={context.last_retrieved_items || []} />
          )}
        </div>
      ))}
      <div ref={messagesEndRef} />
    </div>
  );
}

export default Chatbot;