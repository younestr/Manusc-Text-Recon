import React, { useState } from 'react';
import axios from 'axios';
import './styles.css';  // Import the CSS file

function App() {
  const [image, setImage] = useState(null);
  const [transcription, setTranscription] = useState('');

  const handleFileChange = (e) => {
    setImage(e.target.files[0]);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!image) {
      alert('Please select an image');
      return;
    }

    // Create a FormData object
    const formData = new FormData();
    formData.append('image', image);

    try {
      const response = await axios.post(
        'https://bfb6-34-87-196-141.ngrok-free.app/transcribe',
        formData,
        { headers: { 'Content-Type': 'multipart/form-data' } }
      );
      setTranscription(response.data.transcription);
    } catch (error) {
      console.error('Error uploading image:', error);
    }
  };

  return (
    <div className="App">
      <h1>Image Text Recognition</h1>
      <form onSubmit={handleSubmit}>
        <input type="file" accept="image/*" onChange={handleFileChange} />
        <button type="submit">Submit</button>
      </form>
      {transcription && (
        <div>
          <h2>Transcription:</h2>
          <p>{transcription}</p>
        </div>
      )}
    </div>
  );
}

export default App;
