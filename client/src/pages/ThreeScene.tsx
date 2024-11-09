import React, { useRef, useEffect, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/Addons.js';
import axios from "axios";

const ThreeScene: React.FC = () => {
    const mountRef = useRef<HTMLDivElement | null>(null);
    const [file, setFile] = useState<File | null>(null);

    const createCube = (x: number, y: number, z: number) => {
        const geometry = new THREE.BoxGeometry(10, 10, 10); // Adjust dimensions to match room size
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

    const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
        event.preventDefault();
        if (!file) return;

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await axios.post('http://127.0.0.1:8080/api/floorplan', formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
            });
            console.log(response.data);

            // Check if response data has a lines property
            if (response.data && Array.isArray(response.data.lines)) {
                const building = createBuilding(response.data.lines); // Pass the lines array to createBuilding

                // Add the building to the scene
                const scene = new THREE.Scene();
                scene.add(building);
                scene.add(new THREE.AmbientLight(0xffffff, 0.5));
                const pointLight = new THREE.PointLight(0xffffff, 1);
                pointLight.position.set(20, 50, 20);
                scene.add(pointLight);

                const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
                camera.position.z = 120;
                camera.position.y = 50;

                const renderer = new THREE.WebGLRenderer();
                renderer.setSize(window.innerWidth, window.innerHeight);
                if (mountRef.current) {
                    mountRef.current.appendChild(renderer.domElement);
                }

                const controls = new OrbitControls(camera, renderer.domElement);
                controls.enableDamping = true;
                controls.dampingFactor = 0.25;
                controls.enableZoom = true;
                controls.screenSpacePanning = false;
                controls.maxPolarAngle = Math.PI / 2;

                const animate = () => {
                    requestAnimationFrame(animate);
                    controls.update();
                    renderer.render(scene, camera);
                };
                animate();
            } else {
                console.error('Unexpected response data format:', response.data);
            }
        } catch (error) {
            console.error('Error uploading file:', error);
        }
    };

    const createWall = (start: [number, number], end: [number, number], height: number) => {
        const length = Math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2);
        const geometry = new THREE.BoxGeometry(length, height, 0.1);
        const material = new THREE.MeshBasicMaterial({ color: 0x808080 });
        const wall = new THREE.Mesh(geometry, material);
        wall.position.set((start[0] + end[0]) / 2, height / 2, (start[1] + end[1]) / 2);
        const angle = Math.atan2(end[1] - start[1], end[0] - start[0]);
        wall.rotation.y = -angle;
        return wall;
    };

    const createBuilding = (lines: [number, number, number, number][]) => {
        const building = new THREE.Group();

        lines.forEach((wallData: [number, number, number, number]) => {
            const [x1, y1, x2, y2] = wallData;
            const wall = createWall([x1, y1], [x2, y2], 10); // Set wall height
            building.add(wall);
        });

        return building;
    };

    useEffect(() => {
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        const renderer = new THREE.WebGLRenderer();
        renderer.setSize(window.innerWidth, window.innerHeight);
        if (mountRef.current) {
            mountRef.current.appendChild(renderer.domElement);
        }

        const controls = new OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.25;
        controls.enableZoom = true;
        controls.screenSpacePanning = false;
        controls.maxPolarAngle = Math.PI / 2;

        const animate = () => {
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        };
        animate();

        return () => {
            if (mountRef.current) {
                mountRef.current.removeChild(renderer.domElement);
            }
        };
    }, []);

    return (
        <div>
            <form onSubmit={handleSubmit}>
                <input type="file" onChange={handleFileChange} />
                <button type="submit">Upload</button>
            </form>
            <div ref={mountRef}></div>
        </div>
    );
};

export default ThreeScene;