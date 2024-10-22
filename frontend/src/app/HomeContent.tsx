"use client";

import { useState, useEffect } from 'react';
import io from 'socket.io-client';
import styles from "./page.module.css";
import Image from 'next/image';

const socket = io('http://localhost:5500', {
    withCredentials: true,
    transports: ['websocket']
});

export default function Home() {
  const [inputText, setInputText] = useState('');
  const [numSequences, setNumSequences] = useState('80');
  const [generatedText, setGeneratedText] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);

  useEffect(() => {
    socket.on('new_word', (data) => {
      setGeneratedText(prev => prev + ' ' + data.word);
    });

    socket.on('generation_complete', () => {
      setIsGenerating(false);
    });

    return () => {
      socket.off('new_word');
      socket.off('generation_complete');
    };
  }, []);

  const handleSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setGeneratedText('');
    setIsGenerating(true);
    const num = parseInt(numSequences) || 80;
    socket.emit('generate_text', { input_sequence: inputText, num_words: num });
  };

  return (
    <div className={styles.page}>
      <main className={styles.main}>
        <h2>INFO6106: Neural Networks - Assignment 05</h2>
        <h4>RNN - Implemented in pure python</h4>
        <h4><i>Book: Stories From Tagore</i></h4>
        <h6><i>Source: <a href="https://www.gutenberg.org/cache/epub/33525/pg33525-images.html">Stories From Tagore</a></i></h6>
        <div className={styles.imageContainer}>
          <Image
            src="https://www.gutenberg.org/cache/epub/33525/images/cover.jpg"
            alt="Stories From Tagore Book Cover"
            width={100}
            height={150}
            style={{ width: '100%', height: 'auto' }}
          />
        </div>
        <form onSubmit={handleSubmit} className={styles.form}>
          <input
            type="number"
            value={numSequences}
            onChange={(e) => setNumSequences(e.target.value)}
            placeholder="Number of sequences (default: 80)"
            className={styles.input}
          />
          <textarea
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            placeholder="Enter seed text..."
            className={styles.textarea}
          />
          <button type="submit" disabled={isGenerating} className={styles.button}>
            {isGenerating ? 'Generating...' : 'Generate Text'}
          </button>
        </form>
        
        {generatedText && (
          <div className={styles.generatedText}>
            <h3>Generated Text:</h3>
            <p>{generatedText}</p>
          </div>
        )}
      </main>
    </div>
  );
}