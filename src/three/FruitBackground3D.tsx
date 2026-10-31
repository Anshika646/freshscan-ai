import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js';
import { SlidersHorizontal, Pause, Play, RotateCcw, ChevronDown } from 'lucide-react';
import {
  buildAllBotanicalFruitActors,
  createDisposableRegistry,
  createOpticalCalibratorGroup,
  disposeRegistry,
} from './proceduralFruits';
import type { SupportedFruitId } from '../types';

interface FruitBackground3DProps {
  scrollProgressRef: React.MutableRefObject<number>;
  focusedFruit: SupportedFruitId | null;
  onSelectFocusFruit: (fruit: SupportedFruitId | null) => void;
  isScanning: boolean;
  isOverviewActive: boolean;
}

const DOCK_FRUITS: Array<{ id: SupportedFruitId | null; label: string }> = [
  { id: null, label: 'All Orbit' },
  { id: 'apple', label: 'Apple' },
  { id: 'banana', label: 'Banana' },
  { id: 'orange', label: 'Orange' },
  { id: 'strawberry', label: 'Strawberry' },
  { id: 'grape', label: 'Grape' },
  { id: 'pomegranate', label: 'Pomegranate' },
  { id: 'guava', label: 'Guava' },
];

export const FruitBackground3D: React.FC<FruitBackground3DProps> = ({
  scrollProgressRef,
  focusedFruit,
  onSelectFocusFruit,
  isScanning,
  isOverviewActive,
}) => {
  const mountRef = useRef<HTMLDivElement | null>(null);
  const [webglAvailable, setWebglAvailable] = useState(true);
  const [orbitPaused, setOrbitPaused] = useState(false);
  const [dockOpen, setDockOpen] = useState(false);

  const focusedFruitRef = useRef<SupportedFruitId | null>(focusedFruit);
  const isScanningRef = useRef<boolean>(isScanning);
  const isOverviewRef = useRef<boolean>(isOverviewActive);
  const orbitPausedRef = useRef<boolean>(orbitPaused);
  const resetImpulseRef = useRef<boolean>(false);

  useEffect(() => {
    focusedFruitRef.current = focusedFruit;
  }, [focusedFruit]);

  useEffect(() => {
    isScanningRef.current = isScanning;
  }, [isScanning]);

  useEffect(() => {
    isOverviewRef.current = isOverviewActive;
  }, [isOverviewActive]);

  useEffect(() => {
    orbitPausedRef.current = orbitPaused;
  }, [orbitPaused]);

  useEffect(() => {
    const container = mountRef.current;
    if (!container) return;

    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({
        antialias: true,
        alpha: false,
        powerPreference: 'high-performance',
      });
    } catch {
      setWebglAvailable(false);
      return;
    }

    const gl = renderer.getContext();
    const dbg = gl.getExtension('WEBGL_debug_renderer_info');
    const rendererInfo = dbg ? String(gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL)) : '';
    const isSoftwareGl = /SwiftShader|llvmpipe|Software/i.test(rendererInfo);

    renderer.setPixelRatio(isSoftwareGl ? 1 : Math.min(window.devicePixelRatio, 2));
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.08;
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.domElement.style.display = 'block';
    renderer.domElement.style.width = '100%';
    renderer.domElement.style.height = '100%';
    container.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    const bgHex = 0x070a08;
    scene.background = new THREE.Color(bgHex);
    scene.fog = new THREE.FogExp2(bgHex, 0.055);

    const camera = new THREE.PerspectiveCamera(
      42,
      window.innerWidth / Math.max(1, window.innerHeight),
      0.1,
      60,
    );
    camera.position.set(0, 0.15, 7.6);

    // Studio IBL reflections via RoomEnvironment + PMREMGenerator
    const pmremGenerator = new THREE.PMREMGenerator(renderer);
    const roomEnv = new RoomEnvironment();
    const envTarget = pmremGenerator.fromScene(roomEnv, 0.04);
    scene.environment = envTarget.texture;
    roomEnv.dispose();
    pmremGenerator.dispose();

    // Three-point botanical studio lighting
    const hemiLight = new THREE.HemisphereLight(0xdbeafe, 0x09120d, 0.55);
    scene.add(hemiLight);

    const keyLight = new THREE.DirectionalLight(0xfff4e6, 2.25);
    keyLight.position.set(5.5, 6.2, 5.8);
    scene.add(keyLight);

    const emeraldRimLight = new THREE.DirectionalLight(0x10b981, 1.35);
    emeraldRimLight.position.set(-5.2, 2.4, -3.8);
    scene.add(emeraldRimLight);

    const warmFillLight = new THREE.DirectionalLight(0xfde68a, 0.48);
    warmFillLight.position.set(-3.2, -3.5, 4.2);
    scene.add(warmFillLight);

    // Build composite 3D botanical actors & precision calibrator ring
    const registry = createDisposableRegistry();
    const actors = buildAllBotanicalFruitActors(registry);
    const stageGroup = new THREE.Group();
    scene.add(stageGroup);

    actors.forEach((actor) => {
      actor.group.position.copy(actor.homePos);
      actor.group.scale.setScalar(actor.baseScale);
      stageGroup.add(actor.group);
    });

    const calibrator = createOpticalCalibratorGroup(registry);
    calibrator.group.position.set(2.35, 0.32, 0.25);
    calibrator.group.rotation.set(0.18, -0.32, 0);
    stageGroup.add(calibrator.group);

    // Subtle spatial depth dust points (adds 3D parallax reference on scroll)
    const sporeCount = isSoftwareGl ? 48 : 110;
    const sporePositions = new Float32Array(sporeCount * 3);
    for (let i = 0; i < sporeCount; i++) {
      sporePositions[i * 3] = ((i * 73) % 100) / 100 * 14 - 7;
      sporePositions[i * 3 + 1] = ((i * 41) % 100) / 100 * 12 - 6;
      sporePositions[i * 3 + 2] = -(((i * 29) % 100) / 100 * 7) - 1.2;
    }
    const sporeGeo = new THREE.BufferGeometry();
    sporeGeo.setAttribute('position', new THREE.BufferAttribute(sporePositions, 3));
    registry.geometries.push(sporeGeo);
    const sporeMat = new THREE.PointsMaterial({
      color: 0x34d399,
      size: 0.032,
      sizeAttenuation: true,
      transparent: true,
      opacity: 0.26,
    });
    registry.materials.push(sporeMat);
    const sporePoints = new THREE.Points(sporeGeo, sporeMat);
    scene.add(sporePoints);

    // Pre-allocated scratch math vectors (zero per-frame heap allocations)
    const targetPos = new THREE.Vector3();
    const focalSlot = new THREE.Vector3(2.35, 0.22, 0.55);
    const camTargetPos = new THREE.Vector3(0, 0.15, 7.6);
    const camLookTarget = new THREE.Vector3(0.35, 0.0, 0);
    const currentLookTarget = new THREE.Vector3(0.35, 0.0, 0);
    const targetScaleVec = new THREE.Vector3();

    const pointer = { x: 0, y: 0 };
    const smoothedPointer = { x: 0, y: 0 };
    let smoothedScroll = scrollProgressRef.current;

    const handlePointerMove = (e: PointerEvent) => {
      pointer.x = (e.clientX / Math.max(1, window.innerWidth)) * 2 - 1;
      pointer.y = -((e.clientY / Math.max(1, window.innerHeight)) * 2 - 1);
    };

    const handleResize = () => {
      const w = window.innerWidth;
      const h = Math.max(1, window.innerHeight);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
      renderer.setPixelRatio(isSoftwareGl ? 1 : Math.min(window.devicePixelRatio, 2));
    };

    window.addEventListener('pointermove', handlePointerMove, { passive: true });
    window.addEventListener('resize', handleResize);

    let disposed = false;
    let rafId = 0;

    function afterDraw(cb: () => void): void {
      if (!(gl instanceof WebGL2RenderingContext)) {
        setTimeout(cb, 16);
        return;
      }
      const sync = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0);
      gl.flush();
      const poll = () => {
        if (disposed || !sync) return;
        const status = gl.clientWaitSync(sync, 0, 0);
        if (status === gl.ALREADY_SIGNALED || status === gl.CONDITION_SATISFIED) {
          gl.deleteSync(sync);
          cb();
        } else {
          setTimeout(poll, 4);
        }
      };
      setTimeout(poll, 4);
    }

    let prevTime = performance.now();
    let elapsedOrbitTime = 0;

    const tick = () => {
      if (disposed) return;

      const now = performance.now();
      const dt = Math.min((now - prevTime) / 1000, 0.05);
      prevTime = now;

      if (!prefersReducedMotion && !orbitPausedRef.current) {
        elapsedOrbitTime += dt;
      }

      if (resetImpulseRef.current) {
        pointer.x = 0;
        pointer.y = 0;
        resetImpulseRef.current = false;
      }

      const damp = 1 - Math.exp(-5.0 * dt);
      const rawScroll = isOverviewRef.current ? scrollProgressRef.current : 0.22;
      smoothedScroll = THREE.MathUtils.lerp(smoothedScroll, rawScroll, damp);
      smoothedPointer.x = THREE.MathUtils.lerp(smoothedPointer.x, pointer.x, damp);
      smoothedPointer.y = THREE.MathUtils.lerp(smoothedPointer.y, pointer.y, damp);

      const isMobileViewport = window.innerWidth < 900;
      const focusId = focusedFruitRef.current;
      const scanningActive = isScanningRef.current;

      // Scroll-choreographed camera dolly & subtle vertical glide
      const scrollAngle = smoothedScroll * Math.PI * 0.85;
      camTargetPos.set(
        smoothedPointer.x * 0.42 + Math.sin(scrollAngle) * 0.55,
        0.15 + smoothedPointer.y * 0.28 - smoothedScroll * 0.75,
        (isMobileViewport ? 9.2 : 7.55) + Math.sin(smoothedScroll * Math.PI) * 0.85,
      );
      camera.position.lerp(camTargetPos, damp);

      camLookTarget.set(
        isMobileViewport ? 0 : 0.45 - smoothedScroll * 0.65,
        -smoothedScroll * 0.35,
        0,
      );
      currentLookTarget.lerp(camLookTarget, damp);
      camera.lookAt(currentLookTarget);

      // Update each composite 3D botanical fruit along its scroll + orbital trajectory
      focalSlot.set(isMobileViewport ? 0 : 2.35, isMobileViewport ? 0.9 : 0.22, 0.55);

      let calibratorAnchorActor = actors[0];

      for (let i = 0; i < actors.length; i++) {
        const actor = actors[i];
        const isFocused = focusId === actor.id || (focusId === 'jujube' && actor.id === 'apple');

        if (isFocused) {
          calibratorAnchorActor = actor;
          targetPos.copy(focalSlot);
          targetPos.y += Math.sin(elapsedOrbitTime * 1.4) * 0.08;
        } else {
          // Scroll-linked 3D orbital choreography + gentle harmonic bobbing
          const s = smoothedScroll;
          const swirlAngle = s * 1.45 + actor.floatPhase * 0.25;
          const cosA = Math.cos(swirlAngle * 0.35);
          const sinA = Math.sin(swirlAngle * 0.35);

          const baseX = actor.homePos.x + actor.scrollDrift.x * s;
          const baseY =
            actor.homePos.y +
            actor.scrollDrift.y * Math.sin(s * Math.PI * 0.92) +
            Math.sin(elapsedOrbitTime * 1.05 + actor.floatPhase) * 0.14;
          const baseZ = actor.homePos.z + actor.scrollDrift.z * Math.sin(s * Math.PI);

          targetPos.set(
            baseX * cosA - baseZ * sinA * 0.22,
            baseY,
            baseZ + (focusId ? -0.9 : 0) + (!isOverviewRef.current ? -1.15 : 0),
          );

          // If focusId is null and Apple is in hero position, push other fruits cleanly clear
          if (focusId && actor.id === 'apple') {
            targetPos.set(-2.2, 1.85, -1.9);
          }
        }

        actor.group.position.lerp(targetPos, damp);

        const desiredScale =
          actor.baseScale *
          (isFocused ? 1.24 : focusId ? 0.82 : 1.0) *
          (!isOverviewRef.current ? 0.88 : 1.0);
        targetScaleVec.setScalar(desiredScale);
        actor.group.scale.lerp(targetScaleVec, damp);

        if (!prefersReducedMotion && !orbitPausedRef.current) {
          const speedBoost = scanningActive && (isFocused || (!focusId && i === 0)) ? 2.6 : 1.0;
          actor.group.rotation.x += actor.rotSpeed.x * dt * speedBoost;
          actor.group.rotation.y +=
            actor.rotSpeed.y * dt * speedBoost + (sDeltaRotation(smoothedScroll) * 0.002);
          actor.group.rotation.z += actor.rotSpeed.z * dt * speedBoost;
        }
      }

      // Precision optical calibrator ring tracks the focal hero specimen
      calibrator.group.position.lerp(calibratorAnchorActor.group.position, damp * 1.2);
      calibrator.group.rotation.y =
        -0.32 + smoothedPointer.x * 0.25 + Math.sin(elapsedOrbitTime * 0.6) * 0.14;
      calibrator.group.rotation.x = 0.18 - smoothedPointer.y * 0.18 + smoothedScroll * 0.45;
      calibrator.innerRing.rotation.z = elapsedOrbitTime * (scanningActive ? 2.4 : 0.45);

      const targetReticleOpacity = !isOverviewRef.current && !scanningActive ? 0.14 : scanningActive ? 0.72 : 0.34;
      calibrator.reticleMat.opacity = THREE.MathUtils.lerp(
        calibrator.reticleMat.opacity,
        targetReticleOpacity,
        damp,
      );

      sporePoints.rotation.y = elapsedOrbitTime * 0.03 + smoothedScroll * 0.35;
      sporePoints.position.y = smoothedScroll * 1.1;

      renderer.render(scene, camera);

      if (prefersReducedMotion) {
        // Render once on state changes or resize without continuous motion
        rafId = window.setTimeout(tick, 250);
      } else if (isSoftwareGl) {
        afterDraw(() => {
          if (!disposed) rafId = requestAnimationFrame(tick);
        });
      } else {
        rafId = requestAnimationFrame(tick);
      }
    };

    function sDeltaRotation(s: number): number {
      return Math.sin(s * Math.PI * 2);
    }

    rafId = requestAnimationFrame(tick);

    return () => {
      disposed = true;
      cancelAnimationFrame(rafId);
      clearTimeout(rafId);
      window.removeEventListener('pointermove', handlePointerMove);
      window.removeEventListener('resize', handleResize);
      envTarget.dispose();
      disposeRegistry(registry);
      renderer.dispose();
      if (renderer.domElement.parentNode === container) {
        container.removeChild(renderer.domElement);
      }
    };
  }, [scrollProgressRef]);

  return (
    <>
      {/* Full-Viewport Fixed 3D WebGL Background Layer */}
      <div
        ref={mountRef}
        aria-hidden="true"
        className="fixed inset-0 w-screen h-screen z-0 pointer-events-none"
      >
        {!webglAvailable && (
          <div className="w-full h-full bg-[radial-gradient(circle_at_72%_32%,#0D261B_0%,#070A08_62%)]" />
        )}
      </div>

      {/* Measured Readability Vignette Over 3D Scene (Zero backdrop-filter for GPU frame pacing) */}
      <div
        aria-hidden="true"
        className="fixed inset-0 z-[1] pointer-events-none bg-[radial-gradient(circle_at_75%_38%,rgba(7,10,8,0.22)_0%,rgba(7,10,8,0.74)_62%,rgba(7,10,8,0.92)_100%)]"
      />

      {/* Collapsible Bottom-Right 3D Scene Dock (<= 10% viewport footprint, solid dark surface) */}
      {webglAvailable && (
        <aside
          aria-label="3D Botanical Scene Controls"
          className="fixed bottom-4 right-4 z-30 hidden sm:block"
        >
          {!dockOpen ? (
            <button
              type="button"
              onClick={() => setDockOpen(true)}
              className="inline-flex items-center gap-2 px-3.5 py-2 text-xs font-medium text-[#D1DDD5] bg-[#0D1310] border border-[#1E2B24] rounded-lg hover:border-[#10B981]/60 hover:text-white transition-colors shadow-lg whitespace-nowrap shrink-0 cursor-pointer"
            >
              <SlidersHorizontal className="w-3.5 h-3.5 text-[#10B981]" />
              <span>3D Produce Stage</span>
            </button>
          ) : (
            <div className="bg-[#0D1310] border border-[#1E2B24] rounded-xl p-3.5 shadow-2xl w-[308px] space-y-3">
              <div className="flex items-center justify-between gap-2">
                <span className="text-xs font-semibold text-[#EEF3EF]">
                  3D Botanical Specimen Stage
                </span>
                <div className="flex items-center gap-1.5">
                  <button
                    type="button"
                    onClick={() => setOrbitPaused((p) => !p)}
                    title={orbitPaused ? 'Resume 3D Orbit' : 'Pause 3D Orbit'}
                    className="p-1.5 text-[#94A89B] hover:text-white bg-[#131C17] border border-[#1E2B24] rounded-md transition-colors cursor-pointer"
                  >
                    {orbitPaused ? <Play className="w-3.5 h-3.5" /> : <Pause className="w-3.5 h-3.5" />}
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      resetImpulseRef.current = true;
                      onSelectFocusFruit(null);
                      setOrbitPaused(false);
                    }}
                    title="Reset 3D Stage"
                    className="p-1.5 text-[#94A89B] hover:text-white bg-[#131C17] border border-[#1E2B24] rounded-md transition-colors cursor-pointer"
                  >
                    <RotateCcw className="w-3.5 h-3.5" />
                  </button>
                  <button
                    type="button"
                    onClick={() => setDockOpen(false)}
                    title="Minimize 3D Controls"
                    className="p-1.5 text-[#94A89B] hover:text-white bg-[#131C17] border border-[#1E2B24] rounded-md transition-colors cursor-pointer"
                  >
                    <ChevronDown className="w-3.5 h-3.5" />
                  </button>
                </div>
              </div>

              <div className="grid grid-cols-4 gap-1.5">
                {DOCK_FRUITS.map((item) => {
                  const active = focusedFruit === item.id;
                  return (
                    <button
                      key={item.label}
                      type="button"
                      onClick={() => onSelectFocusFruit(item.id)}
                      className={`px-2 py-1.5 text-[11px] font-medium rounded-md transition-colors truncate cursor-pointer ${
                        active
                          ? 'bg-[#059669] text-white font-semibold'
                          : 'bg-[#131C17] text-[#94A89B] hover:text-[#EEF3EF]'
                      }`}
                    >
                      {item.label}
                    </button>
                  );
                })}
              </div>
            </div>
          )}
        </aside>
      )}
    </>
  );
};
