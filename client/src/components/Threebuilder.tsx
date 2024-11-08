import * as THREE from "three";
import { useRef, useEffect } from "react";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls";

const Threebuilder = () => {
    const rendererRef = useRef<HTMLDivElement>(null);

    const createCube = (x: number, y: number, z: number) => {
        const geometry = new THREE.BoxGeometry(10, 10, 10);
        const material = new THREE.MeshBasicMaterial({ color: 0x8B0000 });
        const cube = new THREE.Mesh(geometry, material);
        cube.position.set(x, y, z);
        return cube;
    };

    const createBuilding = (floorWidth: number, floorDepth: number, numFloors: number) => {
        const building = new THREE.Group();
        for (let i = 0; i < numFloors; i++) {
            for (let j = 0; j < floorWidth; j++) {
                for (let k = 0; k < floorDepth; k++) {
                    const x = j * 10;
                    const y = i * 10;
                    const z = k * 10;
                    building.add(createCube(x, y, z));
                }
            }
        }
        return building;
    };

    useEffect(() => {
        const renderer = new THREE.WebGLRenderer();
        renderer.setSize(window.innerWidth, window.innerHeight);
        if (rendererRef.current) {
            rendererRef.current.appendChild(renderer.domElement);
        }
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(
            75,
            window.innerWidth / window.innerHeight,
            0.1,
            1000
        );
        camera.position.z = 5;
        const building = createBuilding(5, 3, 10);
        scene.add(building);
        scene.add(new THREE.AmbientLight(0xffffff, 0.5));
        const pointLight = new THREE.PointLight(0xffffff, 1);
        pointLight.position.set(20, 50, 20);
        scene.add(pointLight);

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

    return <div ref={rendererRef}></div>;
};

export default Threebuilder;