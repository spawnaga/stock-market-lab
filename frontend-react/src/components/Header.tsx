import React from 'react';
import { Link } from 'react-router-dom';
import './Header.css';

const Header: React.FC = () => {
  return (
    <header className="header">
      <div className="header-container">
        <h1 className="logo">AI Stock Market Lab</h1>
        <nav className="nav">
          <ul className="nav-list">
            <li className="nav-item">
              <Link to="/" className="nav-link">Dashboard</Link>
            </li>
            <li className="nav-item">
              <Link to="/strategy-lab" className="nav-link">Strategy Lab</Link>
            </li>
            <li className="nav-item">
              <Link to="/agent-monitor" className="nav-link">Agent Monitor</Link>
            </li>
            <li className="nav-item">
              <Link to="/ga-rl-training" className="nav-link">GA+RL Training</Link>
            </li>
          </ul>
        </nav>
      </div>
    </header>
  );
};

export default Header;