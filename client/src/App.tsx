import React, { useState } from "react";
import Navbar from "./components/Navbar";
import Sidebar from "./components/Sidebar";
import ViewWindows from "./components/Viewwindows";

interface FileData {
  name: string;
  url: string;
}

const App: React.FC = () => {
  const [currentView, setCurrentView] = useState<string>("welcome");
  const [floorPlanFiles, setFloorPlanFiles] = useState<FileData[]>([]);
  const [selectedFile, setSelectedFile] = useState<FileData | null>(null);

  const handleViewChange = (view: string) => {
    setCurrentView(view);
    setSelectedFile(null); // Reset the selected file when switching views
  };

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files) {
      const files = Array.from(event.target.files);
      const newFiles = files.map((file) => ({
        name: file.name,
        url: URL.createObjectURL(file),
      }));
      setFloorPlanFiles((prev) => [...prev, ...newFiles]);
      setCurrentView("Floor Plans");
    }
  };

  const handleFileSelect = (file: FileData) => {
    setSelectedFile(file);
    setCurrentView("FilePreview"); // ViewWindows will handle 3D models and images
  };

  const handleFileDelete = (fileToDelete: FileData) => {
    setFloorPlanFiles((prev) =>
      prev.filter((file) => file.name !== fileToDelete.name)
    );

    // Reset selected file if the deleted file is currently selected
    if (selectedFile && selectedFile.name === fileToDelete.name) {
      setSelectedFile(null);
      setCurrentView("Floor Plans");
    }
  };

  return (
    <div style={{ height: "100vh", display: "flex", flexDirection: "column" }}>
      {/* Navbar */}
      <Navbar onNavigate={handleViewChange} />

      {/* Sidebar and Main Content */}
      <div style={{ display: "flex", flex: 1 }}>
        <Sidebar
          onChangeView={handleViewChange}
          floorPlanFiles={floorPlanFiles}
          onFileSelect={handleFileSelect}
          onFileDelete={handleFileDelete}
		  elevatorFile={{
            name: "Test 3D Model",
  			url: "https://modelviewer.dev/shared-assets/models/Astronaut.glb",
          }}
        />
        <ViewWindows
          currentView={currentView}
          onFileUpload={handleFileUpload}
          selectedFile={selectedFile}
        />
      </div>
    </div>
  );
};

export default App;
