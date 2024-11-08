import "./App.css";
import * as THREE from "three";
import { useRef, useEffect } from "react";
import { OrbitControls } from "three/examples/jsm/Addons.js";
import { BrowserRouter as Router, Route, Routes } from "react-router-dom";
import Button from "./components/Button";
import ThreeScene from "./pages/ThreeScene";

const App = () => {
	return (
		<Router>
			<Routes>
				<Route path="/" element={<Button text="Click me"/>} />
				<Route path="/three" element={<ThreeScene />} />
			</Routes>
		</Router>
	)
}
export default App;