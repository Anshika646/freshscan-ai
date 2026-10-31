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
  { id: null, label: 'All 8 Orbit' },
  { id: 'apple', label: 'Apple' },
  { id: 'banana', label: 'Banana' },
  { id: 'orange', label: 'Orange' },
  { id: 'strawberry', label: 'Strawberry' },
  { id: 'grape', label: 'Grape' },
  { id: 'pomegranate', label: 'Pomegranate' },
  { id: 'guava', label: 'Guava' },
  { id: 'jujube', label: 'Jujube' },
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

    const isSoftwareGl = (() => {
      const probeGl = document.createElement('canvas').getContext('webgl2');
      if (!probeGl) return true;
      const ext = probeGl.getExtension('WEBGL_debug_renderer_info');
      const param = ext ? ext.UNMASKED_RENDERER_WEBGL : probeGl.RENDERER;
      const gpu = String(probeGl.getParameter(param));
      probeGl.getExtension('WEBGL_lose_context')?.loseContext();
      return /SwiftShader|llvmpipe|softpipe|Software|Basic Render/i.test(gpu);
    })();

    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({
        antialias: !isSoftwareGl,
        alpha: false,
        powerPreference: 'high-performance',
      });
    } catch {
      setWebglAvailable(false);
      return;
    }

    renderer.setPixelRatio(isSoftwareGl ? 1 : Math.min(window.devicePixelRatio, 1.75));
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.24;
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.domElement.style.display = 'block';
    renderer.domElement.style.width = '100%';
    renderer.domElement.style.height = '100%';
    container.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    // Airy deep botanical slate-charcoal canvas so luminous fruits pop cleanly
    const bgHex = 0x09100c;
    scene.background = new THREE.Color(bgHex);
    scene.fog = new THREE.FogExp2(bgHex, 0.02);

    const camera = new THREE.PerspectiveCamera(
      42,
      window.innerWidth / Math.max(1, window.innerHeight),
      0.1,
      60,
    );
    camera.position.set(0, 0.15, 7.7);

    // Bright studio IBL reflections via RoomEnvironment + PMREMGenerator
    const pmremGenerator = new THREE.PMREMGenerator(renderer);
    const roomEnv = new RoomEnvironment();
    const envTarget = pmremGenerator.fromScene(roomEnv, 0.04);
    scene.environment = envTarget.texture;
    roomEnv.dispose();
    pmremGenerator.dispose();

    // Soft, luminous 3-point + hemisphere studio illumination (keeps fruit undersides light & airy)
    const hemiLight = new THREE.HemisphereLight(0xf0fdf4, 0x1d3328, 1.15);
    scene.add(hemiLight);

    const keyLight = new THREE.DirectionalLight(0xfffbeb, 2.5);
    keyLight.position.set(5.2, 6.5, 6.2);
    scene.add(keyLight);

    const emeraldRimLight = new THREE.DirectionalLight(0x6ee7b7, 1.55);
    emeraldRimLight.position.set(-5.4, 2.6, -3.5);
    scene.add(emeraldRimLight);

    const softBounceLight = new THREE.DirectionalLight(0xfef3c7, 0.85);
    softBounceLight.position.set(-2.5, -4.0, 4.8);
    scene.add(softBounceLight);

    // Build all 8 lightweight composite 3D botanical actors & precision calibrator ring
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
    calibrator.group.position.set(2.25, 0.38, 0.25);
    calibrator.group.rotation.set(0.18, -0.32, 0);
    stageGroup.add(calibrator.group);

    // Featherlight botanical pollen/depth motes for 3D scroll parallax
    const sporeCount = isSoftwareGl ? 36 : 80;
    const sporePositions = new Float32Array(sporeCount * 3);
    for (let i = 0; i < sporeCount; i++) {
      sporePositions[i * 3] = (((i * 73) % 100) / 100) * 14 - 7;
      sporePositions[i * 3 + 1] = (((i * 41) % 100) / 100) * 12 - 6;
      sporePositions[i * 3 + 2] = -((((i * 29) % 100) / 100) * 6.5) - 1.0;
    }
    const sporeGeo = new THREE.BufferGeometry();
    sporeGeo.setAttribute('position', new THREE.BufferAttribute(sporePositions, 3));
    registry.geometries.push(sporeGeo);
    const sporeMat = new THREE.PointsMaterial({
      color: 0x6ee7b7,
      size: 0.03,
      sizeAttenuation: true,
      transparent: true,
      opacity: 0.28,
    });
    registry.materials.push(sporeMat);
    const sporePoints = new THREE.Points(sporeGeo, sporeMat);
    scene.add(sporePoints);

    // Pre-allocated scratch math vectors (zero per-frame heap allocations)
    const targetPos = new THREE.Vector3();
    const focalSlot = new THREE.Vector3(2.25, 0.26, 0.65);
    const camTargetPos = new THREE.Vector3(0, 0.15, 7.7);
    const camLookTarget = new THREE.Vector3(0.3, 0.0, 0);
    const currentLookTarget = new THREE.Vector3(0.3, 0.0, 0);
    const targetScaleVec = new THREE.Vector3();

    const pointer = { x: 0, y: 0 };
    const smoothedPointer = { x: 0, y: 0 };
    let smoothedScroll = scrollProgressRef.current;
    let prevRawScroll = scrollProgressRef.current;
    let scrollVelocity = 0;

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
      renderer.setPixelRatio(isSoftwareGl ? 1 : Math.min(window.devicePixelRatio, 1.75));
    };

    window.addEventListener('pointermove', handlePointerMove, { passive: true });
    window.addEventListener('resize', handleResize);

    let disposed = false;
    let frameId = 0;
    const gl = renderer.getContext();

    function afterDraw(cb: () => void): void {
      if (!(gl instanceof WebGL2RenderingContext)) {
        frameId = window.setTimeout(cb, 80);
        return;
      }
      const sync = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0);
      gl.flush();
      const poll = () => {
        if (disposed || !sync) return;
        const state = gl.getSyncParameter(sync, gl.SYNC_STATUS);
        if (state === gl.UNSIGNALED) {
          frameId = window.setTimeout(poll, 5);
        } else {
          gl.deleteSync(sync);
          frameId = window.setTimeout(cb, 80);
        }
      };
      frameId = window.setTimeout(poll, 5);
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

      const damp = 1 - Math.exp(-5.8 * dt);
      const rawScroll = isOverviewRef.current ? scrollProgressRef.current : 0.2;
      const rawDelta = rawScroll - prevRawScroll;
      prevRawScroll = rawScroll;

      // Capture GSAP scroll velocity so fruits float & spin with a weightless breeze on scroll
      scrollVelocity = THREE.MathUtils.lerp(scrollVelocity, rawDelta / Math.max(dt, 0.008), 0.18);
      smoothedScroll = THREE.MathUtils.lerp(smoothedScroll, rawScroll, damp);
      smoothedPointer.x = THREE.MathUtils.lerp(smoothedPointer.x, pointer.x, damp);
      smoothedPointer.y = THREE.MathUtils.lerp(smoothedPointer.y, pointer.y, damp);

      const isMobileViewport = window.innerWidth < 900;
      const focusId = focusedFruitRef.current;
      const scanningActive = isScanningRef.current;

      // GSAP Scroll-choreographed camera glide
      const scrollArc = smoothedScroll * Math.PI * 1.1;
      camTargetPos.set(
        smoothedPointer.x * 0.45 + Math.sin(scrollArc) * 0.62,
        0.16 + smoothedPointer.y * 0.3 - smoothedScroll * 0.65,
        (isMobileViewport ? 9.1 : 7.65) + Math.sin(smoothedScroll * Math.PI) * 0.7,
      );
      camera.position.lerp(camTargetPos, damp);

      camLookTarget.set(
        isMobileViewport ? 0 : 0.35 - smoothedScroll * 0.55,
        -smoothedScroll * 0.28,
        0,
      );
      currentLookTarget.lerp(camLookTarget, damp);
      camera.lookAt(currentLookTarget);

      // Spotlight slot for whichever fruit is focused by the single-slide GSAP scroll showcase
      focalSlot.set(isMobileViewport ? 0 : 2.35, isMobileViewport ? 0.95 : 0.28, 0.75);

      let calibratorAnchorActor = actors[0];

      for (let i = 0; i < actors.length; i++) {
        const actor = actors[i];
        const isFocused = focusId === actor.id;

        if (isFocused) {
          calibratorAnchorActor = actor;
          targetPos.copy(focalSlot);
          targetPos.y += Math.sin(elapsedOrbitTime * 1.65) * 0.1;
        } else {
          // Weightless 3D GSAP scroll helix + buoyant harmonic float
          const s = smoothedScroll;
          const swirlAngle = s * Math.PI * 1.35 + actor.floatPhase * 0.32;
          const cosA = Math.cos(swirlAngle * 0.45);
          const sinA = Math.sin(swirlAngle * 0.45);

          const baseX = actor.homePos.x + actor.scrollDrift.x * Math.sin(s * Math.PI * 0.95);
          const baseY =
            actor.homePos.y +
            actor.scrollDrift.y * s * 0.65 +
            Math.sin(elapsedOrbitTime * 1.25 + actor.floatPhase) * 0.19 +
            THREE.MathUtils.clamp(scrollVelocity * 0.12, -0.35, 0.35);
          const baseZ = actor.homePos.z + actor.scrollDrift.z * Math.sin(s * Math.PI);

          targetPos.set(
            baseX * cosA - baseZ * sinA * 0.28,
            baseY,
            baseZ + (focusId ? -0.85 : 0) + (!isOverviewRef.current ? -1.0 : 0),
          );

          if (focusId && actor.id === 'apple') {
            targetPos.set(-2.35, 1.75, -1.65);
          }
        }

        actor.group.position.lerp(targetPos, damp);

        const desiredScale =
          actor.baseScale *
          (isFocused ? 1.34 : focusId ? 0.84 : 1.0) *
          (!isOverviewRef.current ? 0.9 : 1.0);
        targetScaleVec.setScalar(desiredScale);
        actor.group.scale.lerp(targetScaleVec, damp);

        if (!prefersReducedMotion && !orbitPausedRef.current) {
          const speedBoost = scanningActive && (isFocused || (!focusId && i === 0)) ? 2.6 : 1.0;
          const scrollSpin = THREE.MathUtils.clamp(scrollVelocity * 1.8, -4.5, 4.5);
          actor.group.rotation.x += (actor.rotSpeed.x * speedBoost + scrollSpin * 0.25) * dt;
          actor.group.rotation.y += (actor.rotSpeed.y * speedBoost + scrollSpin * 0.65) * dt;
          actor.group.rotation.z += actor.rotSpeed.z * speedBoost * dt;
        }
      }

      calibrator.group.position.lerp(calibratorAnchorActor.group.position, damp * 1.25);
      calibrator.group.rotation.y =
        -0.32 + smoothedPointer.x * 0.25 + Math.sin(elapsedOrbitTime * 0.65) * 0.15;
      calibrator.group.rotation.x = 0.18 - smoothedPointer.y * 0.18 + smoothedScroll * 0.45;
      calibrator.innerRing.rotation.z = elapsedOrbitTime * (scanningActive ? 2.4 : 0.5);

      const targetReticleOpacity =
        !isOverviewRef.current && !scanningActive ? 0.12 : scanningActive ? 0.68 : 0.28;
      calibrator.reticleMat.opacity = THREE.MathUtils.lerp(
        calibrator.reticleMat.opacity,
        targetReticleOpacity,
        damp,
      );

      sporePoints.rotation.y = elapsedOrbitTime * 0.035 + smoothedScroll * 0.45;
      sporePoints.position.y = smoothedScroll * 1.25;

      renderer.render(scene, camera);

      if (prefersReducedMotion) {
        frameId = window.setTimeout(tick, 250);
      } else if (isSoftwareGl) {
        afterDraw(tick);
      } else {
        frameId = requestAnimationFrame(tick);
      }
    };

    frameId = requestAnimationFrame(tick);

    return () => {
      disposed = true;
      cancelAnimationFrame(frameId);
      clearTimeout(frameId);
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
          <div className="w-full h-full bg-[radial-gradient(circle_at_72%_32%,#123123_0%,#09100C_68%)]" />
        )}
      </div>

      {/* Whisper-Light Contrast Vignette so Luminous 3D Fruits Stay Crisp & Airy */}
      <div
        aria-hidden="true"
        className="fixed inset-0 z-[1] pointer-events-none bg-[radial-gradient(circle_at_68%_40%,rgba(9,16,12,0.08)_0%,rgba(9,16,12,0.38)_64%,rgba(9,16,12,0.66)_100%)]"
      />

      {/* Collapsible Bottom-Right 3D Scene Dock (<= 10% viewport footprint) */}
      {webglAvailable && (
        <aside
          aria-label="3D Botanical Scene Controls"
          className="fixed bottom-4 right-4 z-30 hidden sm:block"
        >
          {!dockOpen ? (
            <button
              type="button"
              onClick={() => setDockOpen(true)}
              className="inline-flex items-center gap-2 px-3.5 py-2 text-xs font-medium text-[#DCFCE7] bg-[#0E1712] border border-[#24382E] rounded-lg hover:border-[#34D399]/70 hover:text-white transition-colors shadow-lg whitespace-nowrap shrink-0 cursor-pointer"
            >
              <SlidersHorizontal className="w-3.5 h-3.5 text-[#34D399]" />
              <span>3D Produce Stage</span>
            </button>
          ) : (
            <div className="bg-[#0E1712] border border-[#24382E] rounded-xl p-3.5 shadow-2xl w-[312px] space-y-3">
              <div className="flex items-center justify-between gap-2">
                <span className="text-xs font-semibold text-[#F0F7F2]">
                  3D Botanical Specimen Stage
                </span>
                <div className="flex items-center gap-1.5">
                  <button
                    type="button"
                    onClick={() => setOrbitPaused((p) => !p)}
                    title={orbitPaused ? 'Resume 3D Orbit' : 'Pause 3D Orbit'}
                    className="p-1.5 text-[#9EB5A7] hover:text-white bg-[#15211B] border border-[#24382E] rounded-md transition-colors cursor-pointer"
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
                    className="p-1.5 text-[#9EB5A7] hover:text-white bg-[#15211B] border border-[#24382E] rounded-md transition-colors cursor-pointer"
                  >
                    <RotateCcw className="w-3.5 h-3.5" />
                  </button>
                  <button
                    type="button"
                    onClick={() => setDockOpen(false)}
                    title="Minimize 3D Controls"
                    className="p-1.5 text-[#9EB5A7] hover:text-white bg-[#15211B] border border-[#24382E] rounded-md transition-colors cursor-pointer"
                  >
                    <ChevronDown className="w-3.5 h-3.5" />
                  </button>
                </div>
              </div>

              <div className="grid grid-cols-3 gap-1.5">
                {DOCK_FRUITS.map((item) => {
                  const active = focusedFruit === item.id;
                  return (
                    <button
                      key={item.label}
                      type="button"
                      onClick={() => onSelectFocusFruit(item.id)}
                      className={`px-2 py-1.5 text-[11px] font-medium rounded-md border transition-colors whitespace-nowrap truncate cursor-pointer ${
                        active
                          ? 'bg-[#059669] border-[#34D399] text-white'
                          : 'bg-[#131E18] border-[#22352B] text-[#B4C6BB] hover:text-white hover:border-[#345042]'
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
