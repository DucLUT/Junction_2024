import React from "react";
import { Folder, Add, Delete } from "@mui/icons-material";
import "./Sidebar.css";

interface FileData {
  name: string;
  url: string;
}

interface SidebarProps {
  onChangeView: (view: string) => void;
  floorPlanFiles: FileData[];
  onFileSelect: (file: FileData) => void;
  onFileDelete: (file: FileData) => void; // Handle file deletion
  elevatorFile: FileData; // Elevator 3D model passed as a prop
}

const Sidebar: React.FC<SidebarProps> = ({
  onChangeView,
  floorPlanFiles,
  onFileSelect,
  onFileDelete,
  elevatorFile,
}) => {
  return (
    <aside className="sidebar">
      <h2 className="sidebar-title">Project Browser</h2>
      <div className="sidebar-divider"></div>
      <ul className="sidebar-list">
        {/* 3D Views */}
        <li className="sidebar-item" onClick={() => onChangeView("3D Views")}>
          <Folder />
          <span className="sidebar-item-text">3D Views</span>
        </li>

        {/* Floor Plans */}
        <li className="sidebar-item">
          <Folder />
          <span
            className="sidebar-item-text"
            onClick={() => onChangeView("Floor Plans")}
          >
            Floor Plans
          </span>
          <Add
            onClick={() => onChangeView("Upload")}
            style={{ cursor: "pointer", marginLeft: "auto" }}
          />
        </li>

        {/* Uploaded Floor Plans */}
        {floorPlanFiles.map((file, index) => (
          <li key={index} className="sidebar-item">
            <span
              className="sidebar-item-text"
              onClick={() => onFileSelect(file)}
              style={{ cursor: "pointer", flex: 1 }}
            >
              {file.name}
            </span>
            <Delete
              onClick={() => onFileDelete(file)}
              style={{ cursor: "pointer", color: "red", marginLeft: "0.5rem" }}
            />
          </li>
        ))}

        {/* Sections */}
        <li className="sidebar-item">
          <Folder />
          <span
            className="sidebar-item-text"
            onClick={() => onChangeView("Sections")}
          >
            Sections
          </span>
        </li>

        {/* Elevator 3D Model in Sections */}
        <li
          className="sidebar-item"
          onClick={() => onFileSelect(elevatorFile)} // Select elevator file
        >
          <span className="sidebar-item-text" style={{ cursor: "pointer", flex: 1 }}>
            Elevator 3D Model
          </span>
        </li>

        {/* Schedules */}
        <li className="sidebar-item" onClick={() => onChangeView("Schedules")}>
          <Folder />
          <span className="sidebar-item-text">Schedules</span>
        </li>
      </ul>
    </aside>
  );
};

export default Sidebar;
