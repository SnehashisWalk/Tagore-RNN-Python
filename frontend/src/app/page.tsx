// page.tsx
// import HomeContent from './HomeContent';

// export default function Home() {
//   return <HomeContent />;
// }

// page.tsx
// page.tsx
"use client";

import { useState } from 'react';
import HomeContent from './HomeContent';
import FirstPage from './FirstPage';
import SecondPage from './SecondPage';
import ThirdPage from './ThirdPage';
import styles from "./page.module.css";

export default function Home() {
  const [currentPage, setCurrentPage] = useState('home');

  const renderContent = () => {
    switch(currentPage) {
      case 'home':
        return <FirstPage />;
      case 'second':
        return <SecondPage />;
      case 'third':
        return <ThirdPage />;
      default:
        return <HomeContent />;
    }
  };

  return (
    <div className={styles.container}>
      <aside className={styles.sidebar}>
        
        <h2>Home</h2>

        <button className={styles.firstbtn} onClick={() => setCurrentPage('home')}>
          RNN trained on the whole book
        </button>
        <button className={styles.secondbtn} onClick={() => setCurrentPage('second')}>
          RNN trained on the first 3 chapters
        </button>
        <button className={styles.thirdbtn} onClick={() => setCurrentPage('third')}>
          RNN trained on the 1st chapter
        </button>
      </aside>
      <div className={styles.content}>
        {renderContent()}
      </div>
    </div>
  );
}