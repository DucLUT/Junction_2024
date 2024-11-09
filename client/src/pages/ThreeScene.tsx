import React, { useRef, useEffect, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader';
import axios from "axios";

const ThreeScene: React.FC = () => {
    const mountRef = useRef<HTMLDivElement | null>(null);
    const [file, setFile] = useState<File | null>(null);
    const [numFloors, setNumFloors] = useState<number>(1);
    const [height, setHeight] = useState<number>(10);
    const sceneRef = useRef<THREE.Scene | null>(null);
    const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
    const rendererRef = useRef<THREE.WebGLRenderer | null>(null);

    const handleNumFloorsChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        setNumFloors(parseInt(event.target.value, 10));
    };

    const handleHeightChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        setHeight(parseInt(event.target.value, 10));
    };

    const createCube = (x: number, y: number, z: number) => {
        const geometry = new THREE.BoxGeometry(10, 10, 10);
        const material = new THREE.MeshBasicMaterial({ color: 0x8B0000 });
        const cube = new THREE.Mesh(geometry, material);
        cube.position.set(x, y, z);
        return cube;
    };

    const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        if (event.target.files) {
            setFile(event.target.files[0]);
        }
    };

    const handleDrop = (event: React.DragEvent<HTMLDivElement>) => {
        event.preventDefault();
        if (event.dataTransfer.files && event.dataTransfer.files[0]) {
            setFile(event.dataTransfer.files[0]);
        }
    };

    const handleDragOver = (event: React.DragEvent<HTMLDivElement>) => {
        event.preventDefault();
    };

    const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
        event.preventDefault();
        if (!file) return;

        if (file.type === 'image/png') {
            const formData = new FormData();
            formData.append('file', file);

            try {
                const response = await axios.post('http://127.0.0.1:8080/api/floorplan', formData, {
                    headers: {
                        'Content-Type': 'multipart/form-data',
                    },
                });
                console.log(response.data);

                if (response.data && Array.isArray(response.data.lines)) {
                    const building = createBuilding(response.data.lines, numFloors, height);
                    console.log('Building:', building); // Log the building object

                    const scene = sceneRef.current;
                    if (scene) {
                        scene.clear(); // Clear previous objects
                        scene.add(building);
                        scene.add(new THREE.AmbientLight(0xffffff, 0.5));

                        const pointLight = new THREE.PointLight(0xffffff, 1);
                        pointLight.position.set(20, 50, 20);
                        scene.add(pointLight);

                        const gridHelper = new THREE.GridHelper(1000, 100);
                        scene.add(gridHelper);

                        // Calculate the center of the building
                        const center = calculateCenter(response.data.lines);

                        const camera = cameraRef.current;
                        if (camera) {
                            camera.position.set(center.x, center.y + 200, center.z + 500); // Set initial position for better visibility
                            camera.lookAt(center.x, center.y, center.z); // Look at the center of the building
                        }
                    }
                } else {
                    console.error('Unexpected response data format:', response.data);
                }
            } catch (error) {
                console.error('Error uploading file:', error);
            }
        } else if (file.name.endsWith('.obj') || file.name.endsWith('.gltf') || file.name.endsWith('.glb')) {
            const reader = new FileReader();
            reader.onload = (event) => {
                if (event.target?.result) {
                    const scene = sceneRef.current;
                    if (scene) {
                        scene.clear(); // Clear previous objects
                        scene.background = new THREE.Color(0x808080); // Set background color to grey

                        if (file.name.endsWith('.obj')) {
                            const objLoader = new OBJLoader();
                            const object = objLoader.parse(event.target.result as string);
                            scene.add(object);
                        } else if (file.name.endsWith('.gltf') || file.name.endsWith('.glb')) {
                            const gltfLoader = new GLTFLoader();
                            gltfLoader.parse(event.target.result as ArrayBuffer, '', (gltf) => {
                                scene.add(gltf.scene);
                            });
                        }
                    }
                }
            };
            if (file.name.endsWith('.obj')) {
                reader.readAsText(file);
            } else if (file.name.endsWith('.gltf') || file.name.endsWith('.glb')) {
                reader.readAsArrayBuffer(file);
            }
        }
    };

    const createWall = (start: [number, number], end: [number, number], height: number): THREE.Mesh => {
        const length = Math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2);
        const geometry = new THREE.BoxGeometry(length, height, 0.1);
        const material = new THREE.MeshBasicMaterial({ color: 0x505050 }); // Change wall color to a different shade of grey
        const wall = new THREE.Mesh(geometry, material);
        wall.position.set((start[0] + end[0]) / 2, height / 2, (start[1] + end[1]) / 2);
        const angle = Math.atan2(end[1] - start[1], end[0] - start[0]);
        wall.rotation.y = -angle;
        return wall;
    };

    const createFloor = (width: number, depth: number, y: number): THREE.Mesh => {
        const geometry = new THREE.PlaneGeometry(width, depth);
        const material = new THREE.MeshBasicMaterial({ color: 0xAAAAAA, side: THREE.DoubleSide });
        const floor = new THREE.Mesh(geometry, material);
        floor.rotation.x = Math.PI / 2;
        floor.position.y = y;
        return floor;
    };

    const createBuilding = (lines: [number, number, number, number][], numFloors: number, height: number): THREE.Group => {
        const building = new THREE.Group();

        let minX = Infinity, minY = Infinity;
        let maxX = -Infinity, maxY = -Infinity;

        lines.forEach((wallData: [number, number, number, number]) => {
            const [x1, y1, x2, y2] = wallData;
            minX = Math.min(minX, x1, x2);
            minY = Math.min(minY, y1, y2);
            maxX = Math.max(maxX, x1, x2);
            maxY = Math.max(maxY, y1, y2);
        });

        const centerX = (minX + maxX) / 2;
        const centerY = (minY + maxY) / 2;

        for (let i = 0; i < numFloors; i++) {
            lines.forEach((wallData: [number, number, number, number]) => {
                const [x1, y1, x2, y2] = wallData;
                const wall = createWall([x1 - centerX, y1 - centerY], [x2 - centerX, y2 - centerY], height); // Set wall height and shift coordinates
                wall.position.y += i * height; // Adjust wall position for each floor
                building.add(wall);
            });

            // Add a floor to the building
            const floorWidth = maxX - minX;
            const floorDepth = maxY - minY;
            const floor = createFloor(floorWidth, floorDepth, i * height); // Adjust width, depth, and y position as needed
            floor.position.set(0, i * height, 0); // Center the floor at (0, 0)
            building.add(floor);
        }

        return building;
    };

    const calculateCenter = (lines: [number, number, number, number][]) => {
        let minX = Infinity, minY = Infinity, minZ = Infinity;
        let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;

        lines.forEach(([x1, y1, x2, y2]) => {
            minX = Math.min(minX, x1, x2);
            minY = Math.min(minY, y1, y2);
            minZ = Math.min(minZ, 0, 0); // Assuming z is always 0 for 2D floor plans
            maxX = Math.max(maxX, x1, x2);
            maxY = Math.max(maxY, y1, y2);
            maxZ = Math.max(maxZ, 0, 0); // Assuming z is always 0 for 2D floor plans
        });

        return {
            x: (minX + maxX) / 2,
            y: (minY + maxY) / 2,
            z: (minZ + maxZ) / 2
        };
    };

    useEffect(() => {
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x808080); // Set background color to grey
        sceneRef.current = scene;

        const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 3000);
        camera.position.set(0, 200, 500); // Set initial position for better visibility
        cameraRef.current = camera;

        const renderer = new THREE.WebGLRenderer();
        renderer.setSize(window.innerWidth, window.innerHeight);
        rendererRef.current = renderer;
        if (mountRef.current) {
            mountRef.current.appendChild(renderer.domElement);
        }

        const controls = new OrbitControls(camera, renderer.domElement);
        controls.target.set(0, 0, 0); // Center the view on the scene
        controls.enableDamping = true;
        controls.dampingFactor = 0.25;
        controls.enableZoom = true;
        controls.screenSpacePanning = false;
        controls.maxPolarAngle = Math.PI / 1.5; // Allow a top-down view

        const animate = () => {
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        };
        animate();

        const handleResize = () => {
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        };
        window.addEventListener("resize", handleResize);
        return () => {
            window.removeEventListener("resize", handleResize);
        };
    }, []);

    return (
        <div style={{ width: '100%', height: '100%' }}>
            <form onSubmit={handleSubmit}>
                <label>
                    Number of Floors:
                    <input type="number" value={numFloors} onChange={handleNumFloorsChange} />
                </label>
                <label>
                    Height of Floors:
                    <input type="number" value={height} onChange={handleHeightChange} />
                </label>
                <input type="file" onChange={handleFileChange} />
                <button type="submit">Upload</button>
            </form>
            <div
                ref={mountRef}
                style={{ width: '100%', height: '100%' }}
                onDrop={handleDrop}
                onDragOver={handleDragOver}
            ></div>
        </div>
    );
};

export default ThreeScene;