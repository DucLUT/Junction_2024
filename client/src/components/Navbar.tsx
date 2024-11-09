import React from "react";
import "./Navbar.css";

interface NavbarProps {
  onNavigate: (view: string) => void; // Callback for navigation
}

const Navbar: React.FC<NavbarProps> = ({ onNavigate }) => {
  return (
    <header className="navbar">
      <div className="brand">My App</div>
      <ul className="nav-list">
        <li>
          <button
            className="nav-link"
            onClick={() => onNavigate("welcome")}
          >
            Home
          </button>
        </li>
        <li>
          <button
            className="nav-link"
            onClick={() => onNavigate("About")}
          >
            About
          </button>
        </li>
        <li>
          <button
            className="nav-link"
            onClick={() => onNavigate("Contact")}
          >
            Contact
          </button>
        </li>
      </ul>
    </header>
  );
};

export default Navbar;
