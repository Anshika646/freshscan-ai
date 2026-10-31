import * as THREE from 'three';
import type { SupportedFruitId } from '../types';

export interface DisposableRegistry {
  geometries: THREE.BufferGeometry[];
  materials: THREE.Material[];
  textures: THREE.Texture[];
}

export function createDisposableRegistry(): DisposableRegistry {
  return {
    geometries: [],
    materials: [],
    textures: [],
  };
}

export function disposeRegistry(reg: DisposableRegistry): void {
  for (const g of reg.geometries) g.dispose();
  for (const m of reg.materials) m.dispose();
  for (const t of reg.textures) t.dispose();
  reg.geometries.length = 0;
  reg.materials.length = 0;
  reg.textures.length = 0;
}

function trackGeo<T extends THREE.BufferGeometry>(reg: DisposableRegistry, geo: T): T {
  reg.geometries.push(geo);
  return geo;
}

function trackMat<T extends THREE.Material>(reg: DisposableRegistry, mat: T): T {
  reg.materials.push(mat);
  return mat;
}

function trackTex<T extends THREE.Texture>(reg: DisposableRegistry, tex: T): T {
  reg.textures.push(tex);
  return tex;
}

/**
 * Deterministic PRNG so procedural botanical textures look identical across renders.
 */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Lightweight 128x128 luminous botanical peel CanvasTexture (sRGB albedo) + 64x64 linear micro-bump.
 * Kept compact in pixel footprint for fast GPU upload and featherweight 60fps scrolling.
 */
function createPeelTextures(
  reg: DisposableRegistry,
  options: {
    seed: number;
    baseTop: string;
    baseMid: string;
    baseBottom: string;
    stripes?: { color: string; count: number; alpha: number };
    speckles?: { color: string; count: number; maxRadius: number; alpha: number };
    bumpStrength?: number;
  },
): { albedo: THREE.CanvasTexture; bump: THREE.CanvasTexture } {
  const size = 128;
  const rand = mulberry32(options.seed);

  const albedoCanvas = document.createElement('canvas');
  albedoCanvas.width = size;
  albedoCanvas.height = size;
  const ctx = albedoCanvas.getContext('2d')!;

  const grad = ctx.createLinearGradient(0, 0, 0, size);
  grad.addColorStop(0, options.baseTop);
  grad.addColorStop(0.52, options.baseMid);
  grad.addColorStop(1, options.baseBottom);
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, size, size);

  if (options.stripes) {
    ctx.strokeStyle = options.stripes.color;
    ctx.globalAlpha = options.stripes.alpha;
    for (let i = 0; i < options.stripes.count; i++) {
      const x = rand() * size;
      ctx.lineWidth = 0.9 + rand() * 2.2;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.bezierCurveTo(
        x + (rand() - 0.5) * 10,
        size * 0.33,
        x + (rand() - 0.5) * 10,
        size * 0.66,
        x + (rand() - 0.5) * 5,
        size,
      );
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
  }

  if (options.speckles) {
    ctx.fillStyle = options.speckles.color;
    ctx.globalAlpha = options.speckles.alpha;
    for (let i = 0; i < options.speckles.count; i++) {
      const x = rand() * size;
      const y = rand() * size;
      const r = 0.35 + rand() * options.speckles.maxRadius;
      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.globalAlpha = 1;
  }

  const albedo = trackTex(reg, new THREE.CanvasTexture(albedoCanvas));
  albedo.colorSpace = THREE.SRGBColorSpace;
  albedo.wrapS = THREE.RepeatWrapping;
  albedo.wrapT = THREE.ClampToEdgeWrapping;
  albedo.needsUpdate = true;

  // Lightweight 64x64 linear grayscale bump map
  const bumpCanvas = document.createElement('canvas');
  bumpCanvas.width = 64;
  bumpCanvas.height = 64;
  const bCtx = bumpCanvas.getContext('2d')!;
  bCtx.fillStyle = '#808080';
  bCtx.fillRect(0, 0, 64, 64);
  const bumpDots = options.bumpStrength ? Math.floor(140 * options.bumpStrength) : 60;
  for (let i = 0; i < bumpDots; i++) {
    const x = rand() * 64;
    const y = rand() * 64;
    const r = 0.5 + rand() * 1.2;
    const v = Math.floor(105 + rand() * 65);
    bCtx.fillStyle = `rgb(${v},${v},${v})`;
    bCtx.beginPath();
    bCtx.arc(x, y, r, 0, Math.PI * 2);
    bCtx.fill();
  }
  const bump = trackTex(reg, new THREE.CanvasTexture(bumpCanvas));
  bump.wrapS = THREE.RepeatWrapping;
  bump.wrapT = THREE.RepeatWrapping;
  bump.needsUpdate = true;

  return { albedo, bump };
}

/**
 * Shared lightweight curved botanical leaf with bevelled extrusion + midrib vein.
 */
function createBotanicalLeaf(
  reg: DisposableRegistry,
  length: number,
  width: number,
  leafMat: THREE.Material,
  veinMat: THREE.Material,
): THREE.Group {
  const leafGroup = new THREE.Group();

  const shape = new THREE.Shape();
  shape.moveTo(0, 0);
  shape.bezierCurveTo(width * 0.75, length * 0.24, width * 0.85, length * 0.68, 0, length);
  shape.bezierCurveTo(-width * 0.85, length * 0.68, -width * 0.75, length * 0.24, 0, 0);

  const bladeGeo = trackGeo(
    reg,
    new THREE.ExtrudeGeometry(shape, {
      steps: 1,
      depth: 0.012,
      bevelEnabled: true,
      bevelThickness: 0.006,
      bevelSize: 0.006,
      bevelSegments: 1,
    }),
  );

  const pos = bladeGeo.attributes.position;
  for (let i = 0; i < pos.count; i++) {
    const y = pos.getY(i);
    const x = pos.getX(i);
    const t = y / Math.max(0.001, length);
    const archZ = Math.sin(t * Math.PI) * (length * 0.14) - Math.abs(x) * 0.11;
    pos.setZ(i, pos.getZ(i) + archZ);
  }
  bladeGeo.computeVertexNormals();

  const bladeMesh = new THREE.Mesh(bladeGeo, leafMat);
  bladeMesh.rotation.x = -Math.PI * 0.38;
  leafGroup.add(bladeMesh);

  const veinCurve = new THREE.CatmullRomCurve3([
    new THREE.Vector3(0, 0, 0.01),
    new THREE.Vector3(0, length * 0.45, length * 0.13),
    new THREE.Vector3(0, length * 0.92, 0.02),
  ]);
  const veinGeo = trackGeo(reg, new THREE.TubeGeometry(veinCurve, 8, 0.008, 5, false));
  const veinMesh = new THREE.Mesh(veinGeo, veinMat);
  veinMesh.rotation.x = -Math.PI * 0.38;
  leafGroup.add(veinMesh);

  return leafGroup;
}

/**
 * 1. Honeycrisp Apple — Luminous coral-crimson & warm golden blush.
 */
export function createAppleModel(reg: DisposableRegistry): THREE.Group {
  const group = new THREE.Group();
  group.name = 'fruit_apple';

  const { albedo, bump } = createPeelTextures(reg, {
    seed: 101,
    baseTop: '#FB7185',
    baseMid: '#F43F5E',
    baseBottom: '#E11D48',
    stripes: { color: '#FDE047', count: 24, alpha: 0.32 },
    speckles: { color: '#FEF9C3', count: 80, maxRadius: 0.9, alpha: 0.42 },
    bumpStrength: 0.35,
  });

  const profile: THREE.Vector2[] = [];
  const steps = 20;
  for (let i = 0; i <= steps; i++) {
    const t = i / steps;
    const theta = t * Math.PI;
    let r = Math.sin(theta) * 0.76;
    r *= 0.86 + 0.22 * Math.sin(t * Math.PI * 0.85);
    let y = -Math.cos(theta) * 0.72;
    if (t > 0.88) {
      const k = (t - 0.88) / 0.12;
      y -= k * k * 0.15;
    } else if (t < 0.12) {
      const k = (0.12 - t) / 0.12;
      y += k * k * 0.1;
    }
    profile.push(new THREE.Vector2(Math.max(0.001, r), y));
  }

  const bodyGeo = trackGeo(reg, new THREE.LatheGeometry(profile, 28));
  const bPos = bodyGeo.attributes.position;
  for (let i = 0; i < bPos.count; i++) {
    const x = bPos.getX(i);
    const y = bPos.getY(i);
    const z = bPos.getZ(i);
    const angle = Math.atan2(z, x);
    const lobe = 1 + 0.025 * Math.cos(angle * 5);
    bPos.setXYZ(i, x * lobe, y, z * lobe);
  }
  bodyGeo.computeVertexNormals();

  const skinMat = trackMat(
    reg,
    new THREE.MeshStandardMaterial({
      map: albedo,
      bumpMap: bump,
      bumpScale: 0.005,
      roughness: 0.24,
      metalness: 0.02,
      emissive: new THREE.Color(0xf43f5e),
      emissiveIntensity: 0.13,
      envMapIntensity: 1.35,
    }),
  );
  group.add(new THREE.Mesh(bodyGeo, skinMat));

  const stemCurve = new THREE.CatmullRomCurve3([
    new THREE.Vector3(0, 0.52, 0),
    new THREE.Vector3(0.03, 0.73, 0.02),
    new THREE.Vector3(0.1, 0.92, 0.04),
  ]);
  const stemGeo = trackGeo(reg, new THREE.TubeGeometry(stemCurve, 9, 0.028, 6, false));
  const stemMat = trackMat(
    reg,
    new THREE.MeshStandardMaterial({ color: 0x785233, roughness: 0.78 }),
  );
  group.add(new THREE.Mesh(stemGeo, stemMat));

  const leafMat = trackMat(
    reg,
    new THREE.MeshStandardMaterial({
      color: 0x22c55e,
      roughness: 0.36,
      emissive: new THREE.Color(0x16a34a),
      emissiveIntensity: 0.1,
      side: THREE.DoubleSide,
    }),
  );
  const veinMat = trackMat(
    reg,
    new THREE.MeshStandardMaterial({ color: 0x86efac, roughness: 0.48 }),
  );
  const leaf = createBotanicalLeaf(reg, 0.46, 0.21, leafMat, veinMat);
  leaf.position.set(0.05, 0.75, 0.03);
  leaf.rotation.set(0.15, 0.6, -0.55);
  group.add(leaf);

  const calyxGeo = trackGeo(reg, new THREE.ConeGeometry(0.03, 0.08, 5));
  for (let s = 0; s < 5; s++) {
    const a = (s / 5) * Math.PI * 2;
    const sepal = new THREE.Mesh(calyxGeo, stemMat);
    sepal.position.set(Math.cos(a) * 0.055, -0.62, Math.sin(a) * 0.055);
    sepal.rotation.set(Math.PI * 0.82, a, 0);
    group.add(sepal);
  }

  return group;
}

/**
 * 2. Cavendish Banana — Bright sunlit lemon-butter peel with lime crown.
 */
export function createBananaModel(reg: DisposableRegistry): THREE.Group {
  const group = new THREE.Group();
  group.name = 'fruit_banana';

  const { albedo, bump } = createPeelTextures(reg, {
    seed: 202,
    baseTop: '#BEF264',
    baseMid: '#FEF08A',
    baseBottom: '#FACC15',
    speckles: { color: '#A16207', count: 36, maxRadius: 0.8, alpha: 0.2 },
    bumpStrength: 0.2,
  });

  const arc = new THREE.CatmullRomCurve3([
    new THREE.Vector3(-0.76, -0.44, 0.0),
    new THREE.Vector3(-0.35, -0.56, 0.05),
    new THREE.Vector3(0.17, -0.27, 0.04),
    new THREE.Vector3(0.55, 0.25, -0.02),
    new THREE.Vector3(0.65, 0.7, -0.04),
  ]);

  const tubularSegments = 32;
  const radialSegments = 10;
  const bodyGeo = trackGeo(
    reg,
    new THREE.TubeGeometry(arc, tubularSegments, 0.21, radialSegments, false),
  );

  const pos = bodyGeo.attributes.position;
  const centerPt = new THREE.Vector3();
  const offsetVec = new THREE.Vector3();
  for (let i = 0; i <= tubularSegments; i++) {
    const u = i / tubularSegments;
    arc.getPointAt(u, centerPt);
    let taper = Math.sin(Math.pow(u, 0.75) * Math.PI);
    taper = Math.max(0.18, Math.pow(taper, 0.58));
    if (u > 0.86) {
      taper = THREE.MathUtils.lerp(taper, 0.27, (u - 0.86) / 0.14);
    }

    for (let j = 0; j <= radialSegments; j++) {
      const idx = i * (radialSegments + 1) + j;
      offsetVec.set(pos.getX(idx), pos.getY(idx), pos.getZ(idx)).sub(centerPt);
      const radialAngle = (j / radialSegments) * Math.PI * 2;
      const ridge = 1 + 0.07 * Math.cos(radialAngle * 5);
      offsetVec.multiplyScalar(taper * ridge);
      pos.setXYZ(idx, centerPt.x + offsetVec.x, centerPt.y + offsetVec.y, centerPt.z + offsetVec.z);
    }
  }
  bodyGeo.computeVertexNormals();

  const peelMat = trackMat(
    reg,
    new THREE.MeshStandardMaterial({
      map: albedo,
      bumpMap: bump,
      bumpScale: 0.004,
      roughness: 0.34,
      metalness: 0.0,
      emissive: new THREE.Color(0xfde047),
      emissiveIntensity: 0.11,
      envMapIntensity: 1.25,
    }),
  );
  group.add(new THREE.Mesh(bodyGeo, peelMat));

  const pedicelCurve = new THREE.CatmullRomCurve3([
    new THREE.Vector3(0.64, 0.68, -0.04),
    new THREE.Vector3(0.69, 0.88, -0.05),
  ]);
  const pedicelGeo = trackGeo(reg, new THREE.TubeGeometry(pedicelCurve, 6, 0.05, 6, false));
  const woodyMat = trackMat(
    reg,
    new THREE.MeshStandardMaterial({ color: 0x65a30d, roughness: 0.65 }),
  );
  group.add(new THREE.Mesh(pedicelGeo, woodyMat));

  const tipGeo = trackGeo(reg, new THREE.SphereGeometry(0.044, 8, 8));
  const tipMat = trackMat(
    reg,
    new THREE.MeshStandardMaterial({ color: 0x422006, roughness: 0.85 }),
  );
  const tipMesh = new THREE.Mesh(tipGeo, tipMat);
  tipMesh.position.set(-0.78, -0.45, 0);
  group.add(tipMesh);

  return group;
}

/**
 * 3. Navel Orange — Luminous sunlit tangerine peel with fresh mint leaf.
 */
export function createOrangeModel(reg: DisposableRegistry): THREE.Group {
  const group = new THREE.Group();
  group.name = 'fruit_orange';

  const { albedo, bump } = createPeelTextures(reg, {
    seed: 303,
    baseTop: '#FDBA74',
    baseMid: '#FB923C',
    baseBottom: '#F97316',
    speckles: { color: '#EA580C', count: 160, maxRadius: 0.85, alpha: 0.3 },
    bumpStrength: 0.8,
  });

  const profile: THREE.Vector2[] = [];
  const steps = 18;
  for (let i = 0; i <= steps; i++) {
    const t = i / steps;
    const theta = t * Math.PI;
    const r = Math.sin(theta) * 0.76;
    let y = -Math.cos(theta) * 0.7;
    if (t > 0.9) y -= ((t - 0.9) / 0.1) * 0.045;
    if (t < 0.1) y += ((0.1 - t) / 0.1) * 0.045;
    profile.push(new THREE.Vector2(Math.max(0.001, r), y));
  }
  const rindGeo = trackGeo(reg, new THREE.LatheGeometry(profile, 28));
  const rindMat = trackMat(
    reg,
    new THREE.MeshStandardMaterial({
      map: albedo,
      bumpMap: bump,
      bumpScale: 0.012,
      roughness: 0.28,
      metalness: 0.02,
      emissive: new THREE.Color(0xf97316),
      emissiveIntensity: 0.14,
      envMapIntensity: 1.35,
    }),
  );
  group.add(new THREE.Mesh(rindGeo, rindMat));

  const calyxMat = trackMat(
    reg,
    new THREE.MeshStandardMaterial({ color: 0x22c55e, roughness: 0.45 }),
  );
  const buttonGeo = trackGeo(reg, new THREE.CylinderGeometry(0.055, 0.075, 0.045, 10));
  const button = new THREE.Mesh(buttonGeo, calyxMat);
  button.position.set(0, 0.67, 0);
  group.add(button);

  const sepalGeo = trackGeo(reg, new THREE.ConeGeometry(0.028, 0.11, 4));
  for (let i = 0; i < 6; i++) {
    const a = (i / 6) * Math.PI * 2;
    const sepal = new THREE.Mesh(sepalGeo, calyxMat);
    sepal.position.set(Math.cos(a) * 0.085, 0.67, Math.sin(a) * 0.085);
    sepal.rotation.set(Math.PI * 0.48, -a, 0, 'YXZ');
    group.add(sepal);
  }

  const veinMat = trackMat(
    reg,
    new THREE.MeshStandardMaterial({ color: 0xbbf7d0, roughness: 0.48 }),
  );
  const leaf = createBotanicalLeaf(reg, 0.44, 0.19, calyxMat, veinMat);
  leaf.position.set(0.03, 0.69, 0.02);
  leaf.rotation.set(0.2, -0.8, -0.45);
  group.add(leaf);

  return group;
}

/**
 * 4. Garden Strawberry — Crisp rose-scarlet berry with 48 instanced golden seeds.
 */
export function createStrawberryModel(reg: DisposableRegistry): THREE.Group {
  const group = new THREE.Group();
  group.name = 'fruit_strawberry';

  const { albedo, bump } = createPeelTextures(reg, {
    seed: 404,
    baseTop: '#FDA4AF',
    baseMid: '#FB7185',
    baseBottom: '#F43F5E',
    bumpStrength: 0.45,
  });

  const radiusAtT = (t: number): number => {
    const shoulder = Math.sin(Math.pow(t, 0.65) * Math.PI);
    return Math.max(0.002, shoulder * (0.27 + 0.46 * Math.pow(t, 0.55)));
  };
  const yAtT = (t: number): number => -0.66 + t * 1.22;

  const profile: THREE.Vector2[] = [];
  const steps = 20;
  for (let i = 0; i <= steps; i++) {
    const t = i / steps;
    profile.push(new THREE.Vector2(radiusAtT(t), yAtT(t)));
  }

  const berryGeo = trackGeo(reg, new THREE.LatheGeometry(profile, 26));
  const berryMat = trackMat(
    reg,
    new THREE.MeshStandardMaterial({
      map: albedo,
      bumpMap: bump,
      bumpScale: 0.008,
      roughness: 0.25,
      emissive: new THREE.Color(0xf43f5e),
      emissiveIntensity: 0.14,
      envMapIntensity: 1.35,
    }),
  );
  group.add(new THREE.Mesh(berryGeo, berryMat));

  const seedCount = 48;
  const seedGeo = trackGeo(reg, new THREE.SphereGeometry(0.021, 5, 5));
  seedGeo.scale(0.75, 1.35, 0.6);
  const seedMat = trackMat(
    reg,
    new THREE.MeshStandardMaterial({
      color: 0xfef08a,
      roughness: 0.35,
      emissive: new THREE.Color(0xfacc15),
      emissiveIntensity: 0.15,
    }),
  );
  const seedsInstanced = new THREE.InstancedMesh(seedGeo, seedMat, seedCount);
  const dummy = new THREE.Object3D();
  const goldenAngle = Math.PI * (3 - Math.sqrt(5));
  for (let i = 0; i < seedCount; i++) {
    const t = 0.1 + (i / seedCount) * 0.78;
    const r = radiusAtT(t) * 1.006;
    const y = yAtT(t);
    const phi = i * goldenAngle;
    dummy.position.set(Math.cos(phi) * r, y, Math.sin(phi) * r);
    dummy.lookAt(0, y + 0.18, 0);
    dummy.updateMatrix();
    seedsInstanced.setMatrixAt(i, dummy.matrix);
  }
  seedsInstanced.instanceMatrix.needsUpdate = true;
  group.add(seedsInstanced);

  const hullMat = trackMat(
    reg,
    new THREE.MeshStandardMaterial({
      color: 0x4ade80,
      roughness: 0.4,
      emissive: new THREE.Color(0x22c55e),
      emissiveIntensity: 0.1,
      side: THREE.DoubleSide,
    }),
  );
  const veinMat = trackMat(
    reg,
    new THREE.MeshStandardMaterial({ color: 0xbbf7d0, roughness: 0.5 }),
  );
  for (let i = 0; i < 6; i++) {
    const a = (i / 6) * Math.PI * 2;
    const sepal = createBotanicalLeaf(reg, 0.32, 0.12, hullMat, veinMat);
    sepal.position.set(Math.cos(a) * 0.075, 0.54, Math.sin(a) * 0.075);
    sepal.rotation.set(0.35, -a + Math.PI * 0.5, 0, 'YXZ');
    group.add(sepal);
  }

  const stemCurve = new THREE.CatmullRomCurve3([
    new THREE.Vector3(0, 0.52, 0),
    new THREE.Vector3(0.02, 0.7, 0.01),
    new THREE.Vector3(0.06, 0.84, 0.03),
  ]);
  const stemGeo = trackGeo(reg, new THREE.TubeGeometry(stemCurve, 7, 0.024, 5, false));
  group.add(new THREE.Mesh(stemGeo, hullMat));

  return group;
}

/**
 * 5. Grape Cluster — Airy lavender-amethyst berries with velvety bloom.
 */
export function createGrapeClusterModel(reg: DisposableRegistry): THREE.Group {
  const group = new THREE.Group();
  group.name = 'fruit_grape';

  const rachisCurve = new THREE.CatmullRomCurve3([
    new THREE.Vector3(0, -0.52, 0),
    new THREE.Vector3(0.02, 0.14, 0.02),
    new THREE.Vector3(-0.03, 0.72, -0.02),
    new THREE.Vector3(0.07, 0.92, 0.03),
  ]);
  const rachisGeo = trackGeo(reg, new THREE.TubeGeometry(rachisCurve, 12, 0.028, 6, false));
  const stemMat = trackMat(
    reg,
    new THREE.MeshStandardMaterial({ color: 0x84cc16, roughness: 0.65 }),
  );
  group.add(new THREE.Mesh(rachisGeo, stemMat));

  const berryCoords: Array<[number, number, number, number]> = [];
  const tiers = [
    { y: 0.42, count: 5, radius: 0.28, size: 0.19 },
    { y: 0.16, count: 6, radius: 0.32, size: 0.195 },
    { y: -0.1, count: 4, radius: 0.24, size: 0.185 },
    { y: -0.34, count: 3, radius: 0.15, size: 0.175 },
  ];
  tiers.forEach((tier, tIdx) => {
    for (let i = 0; i < tier.count; i++) {
      const a = (i / tier.count) * Math.PI * 2 + tIdx * 0.55;
      berryCoords.push([
        Math.cos(a) * tier.radius,
        tier.y + ((i % 2) - 0.5) * 0.035,
        Math.sin(a) * tier.radius,
        tier.size,
      ]);
    }
  });

  const grapeGeo = trackGeo(reg, new THREE.SphereGeometry(1, 14, 14));
  grapeGeo.scale(0.96, 1.07, 0.96);
  const grapeMat = trackMat(
    reg,
    new THREE.MeshStandardMaterial({
      color: 0xa78bfa,
      roughness: 0.26,
      metalness: 0.04,
      emissive: new THREE.Color(0x8b5cf6),
      emissiveIntensity: 0.18,
      envMapIntensity: 1.4,
    }),
  );
  const berries = new THREE.InstancedMesh(grapeGeo, grapeMat, berryCoords.length);
  const dummy = new THREE.Object3D();
  const colorVar = new THREE.Color();
  berryCoords.forEach(([x, y, z, s], idx) => {
    dummy.position.set(x, y, z);
    dummy.rotation.set(idx * 0.2, idx * 0.5, 0);
    dummy.scale.setScalar(s);
    dummy.updateMatrix();
    berries.setMatrixAt(idx, dummy.matrix);
    colorVar.setHSL(0.75 + (idx % 4) * 0.015, 0.78, 0.64 + (idx % 3) * 0.05);
    berries.setColorAt(idx, colorVar);
  });
  berries.instanceMatrix.needsUpdate = true;
  if (berries.instanceColor) berries.instanceColor.needsUpdate = true;
  group.add(berries);

  const leafMat = trackMat(
    reg,
    new THREE.MeshStandardMaterial({
      color: 0x4ade80,
      roughness: 0.42,
      side: THREE.DoubleSide,
    }),
  );
  const leaf = createBotanicalLeaf(reg, 0.42, 0.22, leafMat, stemMat);
  leaf.position.set(-0.04, 0.68, 0.02);
  leaf.rotation.set(0.25, 2.2, -0.5);
  group.add(leaf);

  return group;
}

/**
 * 6. Ruby Pomegranate — Bright coral-ruby ribbed body with golden crown stamens.
 */
export function createPomegranateModel(reg: DisposableRegistry): THREE.Group {
  const group = new THREE.Group();
  group.name = 'fruit_pomegranate';

  const { albedo, bump } = createPeelTextures(reg, {
    seed: 606,
    baseTop: '#FDA4AF',
    baseMid: '#FB7185',
    baseBottom: '#E11D48',
    speckles: { color: '#FDE68A', count: 95, maxRadius: 1.0, alpha: 0.28 },
    bumpStrength: 0.45,
  });

  const profile: THREE.Vector2[] = [];
  const steps = 18;
  for (let i = 0; i <= steps; i++) {
    const t = i / steps;
    const theta = t * Math.PI;
    const r = Math.sin(theta) * 0.75;
    const y = -Math.cos(theta) * 0.71;
    profile.push(new THREE.Vector2(Math.max(0.002, r), y));
  }
  const bodyGeo = trackGeo(reg, new THREE.LatheGeometry(profile, 28));
  const pos = bodyGeo.attributes.position;
  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i);
    const y = pos.getY(i);
    const z = pos.getZ(i);
    const angle = Math.atan2(z, x);
    const rib = 1 + 0.028 * Math.cos(angle * 6);
    pos.setXYZ(i, x * rib, y, z * rib);
  }
  bodyGeo.computeVertexNormals();

  const rindMat = trackMat(
    reg,
    new THREE.MeshStandardMaterial({
      map: albedo,
      bumpMap: bump,
      bumpScale: 0.009,
      roughness: 0.32,
      emissive: new THREE.Color(0xf43f5e),
      emissiveIntensity: 0.13,
      envMapIntensity: 1.3,
    }),
  );
  group.add(new THREE.Mesh(bodyGeo, rindMat));

  const neckGeo = trackGeo(reg, new THREE.CylinderGeometry(0.15, 0.12, 0.15, 10, 1, true));
  const neckMesh = new THREE.Mesh(neckGeo, rindMat);
  neckMesh.position.set(0, 0.73, 0);
  group.add(neckMesh);

  const lobeGeo = trackGeo(reg, new THREE.ConeGeometry(0.05, 0.16, 4));
  for (let i = 0; i < 6; i++) {
    const a = (i / 6) * Math.PI * 2;
    const lobe = new THREE.Mesh(lobeGeo, rindMat);
    lobe.position.set(Math.cos(a) * 0.14, 0.84, Math.sin(a) * 0.14);
    lobe.rotation.set(-0.38, -a + Math.PI * 0.5, 0, 'YXZ');
    group.add(lobe);
  }

  const stamenGeo = trackGeo(reg, new THREE.CylinderGeometry(0.008, 0.008, 0.11, 4));
  const stamenMat = trackMat(
    reg,
    new THREE.MeshStandardMaterial({
      color: 0xfde047,
      roughness: 0.5,
      emissive: new THREE.Color(0xfbbf24),
      emissiveIntensity: 0.2,
    }),
  );
  for (let i = 0; i < 6; i++) {
    const a = (i / 6) * Math.PI * 2;
    const st = new THREE.Mesh(stamenGeo, stamenMat);
    st.position.set(Math.cos(a) * 0.055, 0.76, Math.sin(a) * 0.055);
    group.add(st);
  }

  return group;
}

/**
 * 7. Tropical Guava — Fresh pastel-lime pyriform fruit with bright leaf.
 */
export function createGuavaModel(reg: DisposableRegistry): THREE.Group {
  const group = new THREE.Group();
  group.name = 'fruit_guava';

  const { albedo, bump } = createPeelTextures(reg, {
    seed: 707,
    baseTop: '#D9F99D',
    baseMid: '#BEF264',
    baseBottom: '#84CC16',
    speckles: { color: '#F7FEE7', count: 90, maxRadius: 0.9, alpha: 0.35 },
    bumpStrength: 0.4,
  });

  const profile: THREE.Vector2[] = [];
  const steps = 20;
  for (let i = 0; i <= steps; i++) {
    const t = i / steps;
    const theta = t * Math.PI;
    const pyriform = 1.07 - 0.24 * t;
    const r = Math.sin(theta) * 0.7 * pyriform;
    const y = -Math.cos(theta) * 0.78;
    profile.push(new THREE.Vector2(Math.max(0.002, r), y));
  }
  const bodyGeo = trackGeo(reg, new THREE.LatheGeometry(profile, 26));
  const skinMat = trackMat(
    reg,
    new THREE.MeshStandardMaterial({
      map: albedo,
      bumpMap: bump,
      bumpScale: 0.008,
      roughness: 0.3,
      emissive: new THREE.Color(0x84cc16),
      emissiveIntensity: 0.12,
      envMapIntensity: 1.3,
    }),
  );
  group.add(new THREE.Mesh(bodyGeo, skinMat));

  const woodyMat = trackMat(
    reg,
    new THREE.MeshStandardMaterial({ color: 0x785233, roughness: 0.8 }),
  );
  const stemCurve = new THREE.CatmullRomCurve3([
    new THREE.Vector3(0, 0.75, 0),
    new THREE.Vector3(-0.04, 0.92, 0.02),
  ]);
  group.add(
    new THREE.Mesh(trackGeo(reg, new THREE.TubeGeometry(stemCurve, 6, 0.024, 5, false)), woodyMat),
  );

  const leafMat = trackMat(
    reg,
    new THREE.MeshStandardMaterial({
      color: 0x4ade80,
      roughness: 0.38,
      side: THREE.DoubleSide,
    }),
  );
  const veinMat = trackMat(
    reg,
    new THREE.MeshStandardMaterial({ color: 0xd9f99d, roughness: 0.48 }),
  );
  const leafA = createBotanicalLeaf(reg, 0.4, 0.18, leafMat, veinMat);
  leafA.position.set(-0.02, 0.8, 0.01);
  leafA.rotation.set(0.2, 0.4, -0.55);
  group.add(leafA);

  return group;
}

/**
 * 8. Crisp Honey Jujube — Glossy jade-to-amber oval exocarp with slender stem & leaf.
 */
export function createJujubeModel(reg: DisposableRegistry): THREE.Group {
  const group = new THREE.Group();
  group.name = 'fruit_jujube';

  const { albedo, bump } = createPeelTextures(reg, {
    seed: 808,
    baseTop: '#D97706',
    baseMid: '#FBBF24',
    baseBottom: '#BEF264',
    speckles: { color: '#92400E', count: 55, maxRadius: 1.1, alpha: 0.25 },
    bumpStrength: 0.22,
  });

  const profile: THREE.Vector2[] = [];
  const steps = 18;
  for (let i = 0; i <= steps; i++) {
    const t = i / steps;
    const theta = t * Math.PI;
    const r = Math.sin(theta) * 0.62;
    const y = -Math.cos(theta) * 0.78;
    profile.push(new THREE.Vector2(Math.max(0.002, r), y));
  }
  const bodyGeo = trackGeo(reg, new THREE.LatheGeometry(profile, 24));
  const skinMat = trackMat(
    reg,
    new THREE.MeshStandardMaterial({
      map: albedo,
      bumpMap: bump,
      bumpScale: 0.004,
      roughness: 0.2,
      metalness: 0.03,
      emissive: new THREE.Color(0xf59e0b),
      emissiveIntensity: 0.13,
      envMapIntensity: 1.4,
    }),
  );
  group.add(new THREE.Mesh(bodyGeo, skinMat));

  const woodyMat = trackMat(
    reg,
    new THREE.MeshStandardMaterial({ color: 0x785233, roughness: 0.78 }),
  );
  const stemCurve = new THREE.CatmullRomCurve3([
    new THREE.Vector3(0, 0.74, 0),
    new THREE.Vector3(0.05, 0.94, 0.02),
  ]);
  group.add(
    new THREE.Mesh(trackGeo(reg, new THREE.TubeGeometry(stemCurve, 6, 0.02, 5, false)), woodyMat),
  );

  const leafMat = trackMat(
    reg,
    new THREE.MeshStandardMaterial({
      color: 0x4ade80,
      roughness: 0.38,
      side: THREE.DoubleSide,
    }),
  );
  const veinMat = trackMat(
    reg,
    new THREE.MeshStandardMaterial({ color: 0xbbf7d0, roughness: 0.48 }),
  );
  const leaf = createBotanicalLeaf(reg, 0.36, 0.16, leafMat, veinMat);
  leaf.position.set(0.03, 0.82, 0.01);
  leaf.rotation.set(0.18, -0.55, -0.48);
  group.add(leaf);

  return group;
}

/**
 * Airy Precision Optical Calibrator Gimbal — Featherlight thin metallic & emerald rings.
 */
export function createOpticalCalibratorGroup(reg: DisposableRegistry): {
  group: THREE.Group;
  innerRing: THREE.Mesh;
  reticleMat: THREE.MeshBasicMaterial;
} {
  const group = new THREE.Group();
  group.name = 'optical_calibrator_gimbal';

  const anodizedMat = trackMat(
    reg,
    new THREE.MeshStandardMaterial({
      color: 0x2b3f34,
      roughness: 0.28,
      metalness: 0.78,
    }),
  );

  const outerGeo = trackGeo(reg, new THREE.TorusGeometry(1.38, 0.011, 10, 72));
  const outerRing = new THREE.Mesh(outerGeo, anodizedMat);
  group.add(outerRing);

  const reticleMat = trackMat(
    reg,
    new THREE.MeshBasicMaterial({
      color: 0x34d399,
      transparent: true,
      opacity: 0.32,
    }),
  );
  const innerGeo = trackGeo(reg, new THREE.TorusGeometry(1.24, 0.005, 8, 72));
  const innerRing = new THREE.Mesh(innerGeo, reticleMat);
  group.add(innerRing);

  const tickGeo = trackGeo(reg, new THREE.BoxGeometry(0.075, 0.012, 0.018));
  for (let i = 0; i < 4; i++) {
    const a = (i / 4) * Math.PI * 2;
    const tick = new THREE.Mesh(tickGeo, reticleMat);
    tick.position.set(Math.cos(a) * 1.31, Math.sin(a) * 1.31, 0);
    tick.rotation.z = a;
    group.add(tick);
  }

  return { group, innerRing, reticleMat };
}

export interface FruitActorSpec {
  id: SupportedFruitId;
  label: string;
  group: THREE.Group;
  homePos: THREE.Vector3;
  scrollDrift: THREE.Vector3;
  rotSpeed: THREE.Vector3;
  floatPhase: number;
  baseScale: number;
}

export function buildAllBotanicalFruitActors(reg: DisposableRegistry): FruitActorSpec[] {
  return [
    {
      id: 'apple',
      label: 'Apple',
      group: createAppleModel(reg),
      homePos: new THREE.Vector3(2.25, 0.38, 0.25),
      scrollDrift: new THREE.Vector3(-0.95, 1.55, -0.85),
      rotSpeed: new THREE.Vector3(0.16, 0.38, 0.09),
      floatPhase: 0.0,
      baseScale: 0.8,
    },
    {
      id: 'banana',
      label: 'Banana',
      group: createBananaModel(reg),
      homePos: new THREE.Vector3(3.85, -1.35, -1.25),
      scrollDrift: new THREE.Vector3(-1.85, 2.75, 1.15),
      rotSpeed: new THREE.Vector3(0.14, 0.3, -0.16),
      floatPhase: 0.9,
      baseScale: 0.78,
    },
    {
      id: 'orange',
      label: 'Orange',
      group: createOrangeModel(reg),
      homePos: new THREE.Vector3(-3.15, 1.45, -1.15),
      scrollDrift: new THREE.Vector3(1.55, -2.35, 1.3),
      rotSpeed: new THREE.Vector3(0.19, -0.32, 0.12),
      floatPhase: 1.8,
      baseScale: 0.74,
    },
    {
      id: 'strawberry',
      label: 'Strawberry',
      group: createStrawberryModel(reg),
      homePos: new THREE.Vector3(0.55, -1.65, 0.45),
      scrollDrift: new THREE.Vector3(-2.15, 2.6, -0.75),
      rotSpeed: new THREE.Vector3(-0.18, 0.42, 0.14),
      floatPhase: 2.7,
      baseScale: 0.68,
    },
    {
      id: 'grape',
      label: 'Grape',
      group: createGrapeClusterModel(reg),
      homePos: new THREE.Vector3(-2.65, -1.5, -0.95),
      scrollDrift: new THREE.Vector3(1.75, 2.9, 0.85),
      rotSpeed: new THREE.Vector3(0.13, 0.28, 0.1),
      floatPhase: 3.6,
      baseScale: 0.74,
    },
    {
      id: 'pomegranate',
      label: 'Pomegranate',
      group: createPomegranateModel(reg),
      homePos: new THREE.Vector3(3.55, 1.85, -1.65),
      scrollDrift: new THREE.Vector3(-1.8, -2.8, 1.45),
      rotSpeed: new THREE.Vector3(-0.15, 0.33, -0.09),
      floatPhase: 4.5,
      baseScale: 0.73,
    },
    {
      id: 'guava',
      label: 'Guava',
      group: createGuavaModel(reg),
      homePos: new THREE.Vector3(-1.15, 2.05, -1.85),
      scrollDrift: new THREE.Vector3(2.35, -2.55, 1.35),
      rotSpeed: new THREE.Vector3(0.16, -0.34, 0.11),
      floatPhase: 5.4,
      baseScale: 0.71,
    },
    {
      id: 'jujube',
      label: 'Jujube',
      group: createJujubeModel(reg),
      homePos: new THREE.Vector3(-3.85, 0.05, -1.55),
      scrollDrift: new THREE.Vector3(2.1, 1.65, 1.15),
      rotSpeed: new THREE.Vector3(0.2, 0.36, -0.12),
      floatPhase: 6.2,
      baseScale: 0.66,
    },
  ];
}
