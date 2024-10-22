"use client";

import { useState, useEffect } from 'react';
import io from 'socket.io-client';
import styles from "./page.module.css";
import Image from 'next/image';

const socket = io('http://localhost:5500', {
    withCredentials: true,
    transports: ['websocket']
});

export default function SecondPage() {

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
        socket.emit('generate_text_3chapters', { input_sequence: inputText, num_words: num });
    };

    return (
        <>
        <div className={styles.pageContent}>
            <div className={styles.pageContentDiv}>
                <div className={styles.pageContentIntroDiv}>
                    <div className={styles.pageContentIntroLeftDiv}>
                        <h1>INFO6106: Neural Networks</h1>
                        <h3>Assignment 05</h3>
                        <h4>Book: Stories from Tagore</h4>
                        <p>Source: <a href="https://www.gutenberg.org/cache/epub/33525/pg33525-images.html" target="_blank">Link</a></p>
                        <p><i>Presented by: Snehashis & Prathana</i></p>
                        <p><i>RNN trained on the first 3 chapters...</i></p>
                    </div>
                    <div className={styles.pageContentIntroRightDiv}>
                        <img src="https://www.gutenberg.org/cache/epub/33525/images/cover.jpg"/>
                    </div>
                </div>
                <div className={styles.pageContentBody}>
                    <form onSubmit={handleSubmit}>
                        <label>Input number of sequences</label>
                        <input
                            type="number"
                            value={numSequences}
                            onChange={(e) => setNumSequences(e.target.value)}
                            placeholder="Number of sequences (default: 80)"
                            className={styles.input}
                        />
                        <label>Input feed sequence</label>
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
                </div>
            </div>
            
            </div>
        </>
    );
}
