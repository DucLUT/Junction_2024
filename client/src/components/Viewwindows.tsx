import React, { useState, useEffect, useRef } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls, useGLTF } from "@react-three/drei";
import "./ViewWindows.css";

interface FileData {
  name: string;
  url: string;
}

interface ViewWindowsProps {
  currentView: string;
  onFileUpload: (event: React.ChangeEvent<HTMLInputElement>) => void;
  selectedFile: FileData | null;
}

// Component to render the 3D model
const Model: React.FC<{ path: string }> = ({ path }) => {
    try {
      console.log("Loading model from path:", path);
      const { scene } = useGLTF(path); // Load the GLTF model
      return <primitive object={scene} scale={0.5} />;
    } catch (error) {
      console.error("Error loading GLTF model:", error);
      return <p>Failed to load model.</p>;
    }
  };
  

const ViewWindows: React.FC<ViewWindowsProps> = ({
  currentView,
  onFileUpload,
  selectedFile,
}) => {
  const [zoom, setZoom] = useState(1); // Track zoom level
  const [offsetX, setOffsetX] = useState(0); // Track horizontal pan
  const [offsetY, setOffsetY] = useState(0); // Track vertical pan
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState<{ x: number; y: number } | null>(
    null
  );

  const containerRef = useRef<HTMLDivElement>(null);

  // Handle zoom with mouse wheel
  const handleWheel = (e: WheelEvent) => {
    e.preventDefault();
    const zoomDelta = e.deltaY > 0 ? -0.1 : 0.1; // Zoom in/out
    setZoom((prevZoom) => Math.max(0.5, Math.min(prevZoom + zoomDelta, 3))); // Limit zoom between 0.5x and 3x
  };

  // Add non-passive event listener for the `wheel` event
  useEffect(() => {
    const container = containerRef.current;
    if (container) {
      container.addEventListener("wheel", handleWheel, { passive: false });
    }
    return () => {
      if (container) {
        container.removeEventListener("wheel", handleWheel);
      }
    };
  }, []);

  // Handle drag start
  const handleMouseDown = (e: React.MouseEvent) => {
    setIsDragging(true);
    setDragStart({ x: e.clientX, y: e.clientY });
  };

  // Handle drag movement
  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDragging || !dragStart) return;
    const deltaX = e.clientX - dragStart.x;
    const deltaY = e.clientY - dragStart.y;
    setOffsetX((prevOffset) => prevOffset + deltaX);
    setOffsetY((prevOffset) => prevOffset + deltaY);
    setDragStart({ x: e.clientX, y: e.clientY });
  };

  // Handle drag end
  const handleMouseUp = () => {
    setIsDragging(false);
    setDragStart(null);
  };

  // Zoom In
  const handleZoomIn = () => {
    setZoom((prevZoom) => Math.min(prevZoom + 0.1, 3)); // Max zoom level: 3x
  };

  // Zoom Out
  const handleZoomOut = () => {
    setZoom((prevZoom) => Math.max(prevZoom - 0.1, 0.5)); // Min zoom level: 0.5x
  };

  return (
    <div className="view-windows">
      {currentView === "welcome" && <h1>Welcome</h1>}

      {currentView === "Upload" && (
        <div className="upload-interface">
          <h1>Upload Your Floor Plans</h1>
          <input
            type="file"
            multiple
            accept="image/*,.gltf,.bin"
            className="upload-input"
            onChange={onFileUpload}
          />
        </div>
      )}

      {/* Render Image or 3D Model */}
      {currentView === "FilePreview" && selectedFile && (
        <div className="file-preview">
          {selectedFile.name.endsWith(".gltf") ? (
            // Render 3D Model Viewer
            <div className="model-container">
              <h1>3D Model Viewer</h1>
              <Canvas style={{ width: "100%", height: "100%" }}>
                <ambientLight intensity={0.8} />
                <directionalLight position={[5, 5, 5]} />
                <Model path={selectedFile.url} />
                <OrbitControls enableZoom={true} />
              </Canvas>
            </div>
          ) : (
            // Render Zoomable Image
            <div
              className="image-viewport"
              ref={containerRef}
              onMouseDown={handleMouseDown}
              onMouseMove={handleMouseMove}
              onMouseUp={handleMouseUp}
              onMouseLeave={handleMouseUp}
              style={{
                cursor: isDragging ? "grabbing" : "grab",
              }}
            >
              {/* Zoom Buttons */}
              <div className="zoom-buttons">
                <button className="zoom-button" onClick={handleZoomIn}>
                  +
                </button>
                <button className="zoom-button" onClick={handleZoomOut}>
                  -
                </button>
              </div>

              {/* Image */}
              <img
                src={selectedFile.url}
                alt={selectedFile.name}
                className="preview-image"
                style={{
                  transform: `scale(${zoom}) translate(${offsetX / zoom}px, ${
                    offsetY / zoom
                  }px)`,
                  transformOrigin: "center center",
                }}
              />
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default ViewWindows;
