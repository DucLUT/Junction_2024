import Threebuilder from "../components/Threebuilder"
import React, {useRef, useEffect} from 'react'
import * as THREE from 'three'
import { OrbitControls } from "three/examples/jsm/Addons.js"

const ThreeScene: React.FC = () => {
    const mountRef = useRef<HTMLDivElement | null>(null);
    const createCube = (x: number, y: number, z: number) => {
        const geometry = new THREE.BoxGeometry(10, 10, 10); // Adjust dimensions to match room size
        const material = new THREE.MeshBasicMaterial({ color: 0x8B0000 });
        const cube = new THREE.Mesh(geometry, material);
        cube.position.set(x, y, z);
        return cube;
      };
      const generateFakeFloorPlans = (numFloors: number) => {
        const floorPlans = [];
      
        // Accurate coordinates for walls, doors, and windows for a single floor
        const singleFloorPlan = {
          walls: [
            // Main living area
            { start: [0, 0], end: [200, 0] },        // Bottom wall
            { start: [200, 0], end: [200, 150] },    // Right wall
            { start: [200, 150], end: [100, 150] },  // Top wall (includes door at 100,150)
            { start: [100, 150], end: [100, 100] },  // Wall segment near door
            { start: [100, 100], end: [0, 100] },    // Top wall to left side
            { start: [0, 100], end: [0, 0] },        // Left wall
      
            // Bedroom
            { start: [100, 0], end: [100, 50] },     // Inner wall separating bedroom
            { start: [100, 50], end: [150, 50] },    // Bottom wall of bedroom
            { start: [150, 50], end: [150, 100] },   // Right wall of bedroom
            { start: [150, 100], end: [100, 100] },  // Top wall of bedroom
      
            // Kitchen
            { start: [0, 100], end: [50, 100] },     // Left wall of kitchen
            { start: [50, 100], end: [50, 150] },    // Top wall of kitchen
            { start: [50, 150], end: [0, 150] },     // Right wall of kitchen
      
            // Bathroom
            { start: [150, 150], end: [200, 150] },  // Bottom wall of bathroom
            { start: [200, 150], end: [200, 200] },  // Right wall of bathroom
            { start: [200, 200], end: [150, 200] },  // Top wall of bathroom
            { start: [150, 200], end: [150, 150] },  // Left wall of bathroom
          ],
          doors: [
            { position: [100, 150] },  // Door to bedroom
            { position: [150, 100] },  // Door to bathroom
          ],
          windows: [
            { start: [50, 150], end: [100, 150] },  // Window along top wall in kitchen
            { start: [0, 50], end: [0, 100] },      // Window along left wall in main living area
          ]
        };
      
        // Create multiple floors with identical floor plan
        for (let i = 0; i < numFloors; i++) {
          floorPlans.push(singleFloorPlan);
        }
      
        return floorPlans;
      };
      

    const createBuilding = (numFloors: number) => {
        const building = new THREE.Group();
        const floorPlans = generateFakeFloorPlans(numFloors);
    
        for (let i = 0; i < numFloors; i++) {
          const floorPlan = floorPlans[i];
          const floor = createFloor(floorPlan.walls, i * 10); // Adjust height as needed
          building.add(floor);
    
          // Add walls for each floor
          floorPlan.walls.forEach((wall: { start: [number, number], end: [number, number] }) => {
            const wallMesh = createWall(wall.start, wall.end, 10); // Set wall height
            wallMesh.position.y += i * 10; // Stack floors vertically
            building.add(wallMesh);
          });
        }
    
        return building;
      };
        //this is old building code but still keep it 
        // for (let i = 0; i < numFloors; i++){
        //     for (let j = 0; j < floorWidth; j++){
        //         for (let k = 0; k < floorDepth; k++){
        //             const x = j * 10;
        //             const y = i * 10;
        //             const z = k * 10;
        //             building.add(createCube(x, y, z));
        //         }
        //     }
        // }
        const createWall = (start: [number, number], end: [number, number], height: number) => {
            const length = Math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2);
            const thickness = 0.5; // Adjust as needed for wall thickness
        
            // Wall geometry with adjusted length and height
            const geometry = new THREE.BoxGeometry(length, height, thickness);
            const material = new THREE.MeshBasicMaterial({ color: 0x8B0000 });
            const wall = new THREE.Mesh(geometry, material);
        
            // Position wall at the midpoint of start and end points
            wall.position.set((start[0] + end[0]) / 2, height / 2, (start[1] + end[1]) / 2);
        
            // Calculate angle and rotate wall to align between start and end points
            const angle = Math.atan2(end[1] - start[1], end[0] - start[0]);
            wall.rotation.y = -angle;
        
            return wall;
          };
          const createFloor = (walls: { start: [number, number], end: [number, number] }[], y: number) => {
            // Find the min and max coordinates based on wall coordinates
            let minX = Infinity, minZ = Infinity;
            let maxX = -Infinity, maxZ = -Infinity;
          
            walls.forEach(({ start, end }) => {
              minX = Math.min(minX, start[0], end[0]);
              maxX = Math.max(maxX, start[0], end[0]);
              minZ = Math.min(minZ, start[1], end[1]);
              maxZ = Math.max(maxZ, start[1], end[1]);
            });
          
            const width = maxX - minX;
            const depth = maxZ - minZ;
          
            const geometry = new THREE.PlaneGeometry(width, depth);
            const material = new THREE.MeshBasicMaterial({ color: 0xAAAAAA, side: THREE.DoubleSide });
            const floor = new THREE.Mesh(geometry, material);
          
            // Position the floor at the correct height (y) and centered based on min/max
            floor.rotation.x = -Math.PI / 2;
            floor.position.set((minX + maxX) / 2, y, (minZ + maxZ) / 2);
          
            return floor;
          };
          



    const setUp = () => {
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x777777);
        const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        camera.position.set(0, 50, 200);
        const renderer = new THREE.WebGLRenderer();
        renderer.setSize(window.innerWidth, window.innerHeight);
        if (mountRef.current) {
            mountRef.current.appendChild(renderer.domElement);
        }
        scene.add(createBuilding(10));
        scene.add(new THREE.AmbientLight(0xffffff, 0.5));
        const pointLight = new THREE.PointLight(0xffffff, 1);
        pointLight.position.set(20,50, 20);
        scene.add(pointLight);
        camera.position.z = 120;
        camera.position.y = 50;
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
        }



    }
    useEffect(setUp, []);
    return (
        <div ref={mountRef}></div>
    )
}
export default ThreeScene