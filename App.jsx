import React from 'react';
import Chatbot from './Chatbot.jsx';
import './App.css';
import './Chatbot.css'; // Import the new Chatbot.css

function App() {
  return (
    <div className="App">
      <h1>Tech Gadget Recommendation Chatbot</h1>
      <Chatbot />
    </div>
  );
}

export default App;