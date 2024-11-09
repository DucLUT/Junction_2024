import React from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls, useGLTF } from "@react-three/drei";

interface FileData {
  name: string;
  url: string;
}

interface ViewWindowsProps {
  currentView: string;
  selectedFile: FileData | null;
}

const Model: React.FC<{ path: string }> = ({ path }) => {
  const { scene } = useGLTF(path); // Load the GLTF model
  return <primitive object={scene} scale={1} />;
};

const ViewWindows: React.FC<ViewWindowsProps> = ({ currentView, selectedFile }) => {
  return (
    <div className="view-windows">
      {currentView === "welcome" && <h1>Welcome</h1>}

      {currentView === "FilePreview" && selectedFile && selectedFile.url.endsWith(".gltf") && (
        <div className="model-container">
          <h1>{selectedFile.name}</h1>
          <Canvas style={{ width: "100%", height: "100%" }}>
            <ambientLight intensity={0.5} />
            <directionalLight position={[10, 10, 5]} />
            <Model path={selectedFile.url} />
            <OrbitControls enableZoom={true} />
          </Canvas>
        </div>
      )}
    </div>
  );
};

export default ViewWindows;
