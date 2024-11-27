import React, { useState } from 'react';
import axios from 'axios';
import './styles.css';  // Import the CSS file

function App() {
  const [image, setImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null); // State for image preview
  const [transcription, setTranscription] = useState('');

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    setImage(file);

    // Generate a temporary URL for the image preview
    if (file) {
      setImagePreview(URL.createObjectURL(file));
    } else {
      setImagePreview(null); // Clear preview if no file is selected
    }
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
        'https://da1c-35-204-140-236.ngrok-free.app/transcribe', // Updated URL
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
      {imagePreview && ( // Display the image preview if available
        <div>
          <h2>Image Preview:</h2>
          <img
            src={imagePreview}
            alt="Preview"
            style={{ maxWidth: '100%', maxHeight: '300px', marginTop: '10px' }}
          />
        </div>
      )}
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
