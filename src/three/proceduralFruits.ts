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
 * Creates a procedural botanical skin CanvasTexture (sRGB albedo) + optional bump map (linear).
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
  const size = 256;
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
      ctx.lineWidth = 1.2 + rand() * 3.8;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.bezierCurveTo(
        x + (rand() - 0.5) * 18,
        size * 0.33,
        x + (rand() - 0.5) * 18,
        size * 0.66,
        x + (rand() - 0.5) * 8,
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
      const r = 0.45 + rand() * options.speckles.maxRadius;
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

  // Grayscale linear bump texture (DO NOT set SRGBColorSpace on bump/data maps)
  const bumpCanvas = document.createElement('canvas');
  bumpCanvas.width = 128;
  bumpCanvas.height = 128;
  const bCtx = bumpCanvas.getContext('2d')!;
  bCtx.fillStyle = '#808080';
  bCtx.fillRect(0, 0, 128, 128);
  const bumpDots = options.bumpStrength ? Math.floor(420 * options.bumpStrength) : 140;
  for (let i = 0; i < bumpDots; i++) {
    const x = rand() * 128;
    const y = rand() * 128;
    const r = 0.6 + rand() * 1.8;
    const v = Math.floor(95 + rand() * 75);
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
 * Shared helper for creating a curved botanical leaf mesh with bevelled extrusion + midrib vein.
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
      steps: 2,
      depth: 0.014,
      bevelEnabled: true,
      bevelThickness: 0.008,
      bevelSize: 0.008,
      bevelSegments: 2,
    }),
  );

  // Subtle organic arch along leaf blade
  const pos = bladeGeo.attributes.position;
  for (let i = 0; i < pos.count; i++) {
    const y = pos.getY(i);
    const x = pos.getX(i);
    const t = y / Math.max(0.001, length);
    const archZ = Math.sin(t * Math.PI) * (length * 0.14) - Math.abs(x) * 0.12;
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
  const veinGeo = trackGeo(reg, new THREE.TubeGeometry(veinCurve, 12, 0.009, 6, false));
  const veinMesh = new THREE.Mesh(veinGeo, veinMat);
  veinMesh.rotation.x = -Math.PI * 0.38;
  leafGroup.add(veinMesh);

  return leafGroup;
}

/**
 * 1. Honeycrisp Apple (5 distinct parts: indented Lathe body, curved woody stem,
 *    extruded botanical leaf + midrib, and 5-sepal basal calyx).
 */
export function createAppleModel(reg: DisposableRegistry): THREE.Group {
  const group = new THREE.Group();
  group.name = 'fruit_apple';

  const { albedo, bump } = createPeelTextures(reg, {
    seed: 101,
    baseTop: '#991B1B',
    baseMid: '#DC2626',
    baseBottom: '#B91C1C',
    stripes: { color: '#F59E0B', count: 38, alpha: 0.22 },
    speckles: { color: '#FDE68A', count: 160, maxRadius: 1.1, alpha: 0.38 },
    bumpStrength: 0.45,
  });

  // Lathe profile with top stem cavity and bottom blossom dimple
  const profile: THREE.Vector2[] = [];
  const steps = 28;
  for (let i = 0; i <= steps; i++) {
    const t = i / steps; // 0 = bottom pole, 1 = top pole
    const theta = t * Math.PI;
    let r = Math.sin(theta) * 0.78;
    // fuller upper shoulders, slightly tapered base
    r *= 0.86 + 0.22 * Math.sin(t * Math.PI * 0.85);
    let y = -Math.cos(theta) * 0.74;
    // indent top stem basin & bottom calyx basin
    if (t > 0.88) {
      const k = (t - 0.88) / 0.12;
      y -= k * k * 0.16;
    } else if (t < 0.12) {
      const k = (0.12 - t) / 0.12;
      y += k * k * 0.11;
    }
    profile.push(new THREE.Vector2(Math.max(0.001, r), y));
  }

  const bodyGeo = trackGeo(reg, new THREE.LatheGeometry(profile, 44));
  // Subtle organic 5-lobed apple asymmetry
  const bPos = bodyGeo.attributes.position;
  for (let i = 0; i < bPos.count; i++) {
    const x = bPos.getX(i);
    const y = bPos.getY(i);
    const z = bPos.getZ(i);
    const angle = Math.atan2(z, x);
    const lobe = 1 + 0.028 * Math.cos(angle * 5);
    bPos.setXYZ(i, x * lobe, y, z * lobe);
  }
  bodyGeo.computeVertexNormals();

  const skinMat = trackMat(
    reg,
    new THREE.MeshPhysicalMaterial({
      map: albedo,
      bumpMap: bump,
      bumpScale: 0.008,
      roughness: 0.26,
      metalness: 0.0,
      clearcoat: 0.55,
      clearcoatRoughness: 0.22,
    }),
  );
  group.add(new THREE.Mesh(bodyGeo, skinMat));

  // Woody curved stem
  const stemCurve = new THREE.CatmullRomCurve3([
    new THREE.Vector3(0, 0.52, 0),
    new THREE.Vector3(0.03, 0.74, 0.02),
    new THREE.Vector3(0.11, 0.94, 0.05),
  ]);
  const stemGeo = trackGeo(reg, new THREE.TubeGeometry(stemCurve, 14, 0.032, 8, false));
  const stemMat = trackMat(
    reg,
    new THREE.MeshStandardMaterial({ color: 0x4a2f1d, roughness: 0.85, metalness: 0.0 }),
  );
  group.add(new THREE.Mesh(stemGeo, stemMat));

  // Botanical leaf attached to upper stem
  const leafMat = trackMat(
    reg,
    new THREE.MeshStandardMaterial({
      color: 0x15803d,
      roughness: 0.42,
      metalness: 0.02,
      side: THREE.DoubleSide,
    }),
  );
  const veinMat = trackMat(
    reg,
    new THREE.MeshStandardMaterial({ color: 0x4ade80, roughness: 0.55 }),
  );
  const leaf = createBotanicalLeaf(reg, 0.48, 0.22, leafMat, veinMat);
  leaf.position.set(0.05, 0.76, 0.03);
  leaf.rotation.set(0.15, 0.6, -0.55);
  group.add(leaf);

  // Basal 5-sepal star calyx
  const calyxGeo = trackGeo(reg, new THREE.ConeGeometry(0.035, 0.09, 5));
  const calyxMat = trackMat(
    reg,
    new THREE.MeshStandardMaterial({ color: 0x3b2a1e, roughness: 0.9 }),
  );
  for (let s = 0; s < 5; s++) {
    const a = (s / 5) * Math.PI * 2;
    const sepal = new THREE.Mesh(calyxGeo, calyxMat);
    sepal.position.set(Math.cos(a) * 0.06, -0.63, Math.sin(a) * 0.06);
    sepal.rotation.set(Math.PI * 0.82, a, 0);
    group.add(sepal);
  }

  return group;
}

/**
 * 2. Cavendish Banana (4 distinct parts: pentagonal tapered curved body, fibrous crown pedicel,
 *    neck collar, and dark floral tip).
 */
export function createBananaModel(reg: DisposableRegistry): THREE.Group {
  const group = new THREE.Group();
  group.name = 'fruit_banana';

  const { albedo, bump } = createPeelTextures(reg, {
    seed: 202,
    baseTop: '#84CC16',
    baseMid: '#FACC15',
    baseBottom: '#EAB308',
    speckles: { color: '#713F12', count: 65, maxRadius: 1.0, alpha: 0.28 },
    bumpStrength: 0.3,
  });

  const arc = new THREE.CatmullRomCurve3([
    new THREE.Vector3(-0.78, -0.45, 0.0),
    new THREE.Vector3(-0.36, -0.58, 0.06),
    new THREE.Vector3(0.18, -0.28, 0.04),
    new THREE.Vector3(0.56, 0.26, -0.02),
    new THREE.Vector3(0.66, 0.72, -0.05),
  ]);

  const tubularSegments = 48;
  const radialSegments = 15;
  const bodyGeo = trackGeo(
    reg,
    new THREE.TubeGeometry(arc, tubularSegments, 0.22, radialSegments, false),
  );

  // Taper ends and sculpt characteristic 5-sided Cavendish banana ridges
  const pos = bodyGeo.attributes.position;
  const centerPt = new THREE.Vector3();
  const offsetVec = new THREE.Vector3();
  for (let i = 0; i <= tubularSegments; i++) {
    const u = i / tubularSegments;
    arc.getPointAt(u, centerPt);
    // Smooth taper at tip (u=0) and stem neck (u=1)
    let taper = Math.sin(Math.pow(u, 0.75) * Math.PI);
    taper = Math.max(0.18, Math.pow(taper, 0.58));
    if (u > 0.86) {
      taper = THREE.MathUtils.lerp(taper, 0.28, (u - 0.86) / 0.14);
    }

    for (let j = 0; j <= radialSegments; j++) {
      const idx = i * (radialSegments + 1) + j;
      offsetVec.set(pos.getX(idx), pos.getY(idx), pos.getZ(idx)).sub(centerPt);
      const radialAngle = (j / radialSegments) * Math.PI * 2;
      const ridge = 1 + 0.075 * Math.cos(radialAngle * 5);
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
      bumpScale: 0.005,
      roughness: 0.44,
      metalness: 0.0,
    }),
  );
  group.add(new THREE.Mesh(bodyGeo, peelMat));

  // Woody crown stem at top (u=1)
  const pedicelCurve = new THREE.CatmullRomCurve3([
    new THREE.Vector3(0.65, 0.7, -0.05),
    new THREE.Vector3(0.7, 0.9, -0.06),
  ]);
  const pedicelGeo = trackGeo(reg, new THREE.TubeGeometry(pedicelCurve, 8, 0.055, 7, false));
  const woodyMat = trackMat(
    reg,
    new THREE.MeshStandardMaterial({ color: 0x4d3b22, roughness: 0.88 }),
  );
  group.add(new THREE.Mesh(pedicelGeo, woodyMat));

  const cutCapGeo = trackGeo(reg, new THREE.CylinderGeometry(0.056, 0.056, 0.03, 8));
  const cutCap = new THREE.Mesh(cutCapGeo, woodyMat);
  cutCap.position.set(0.7, 0.91, -0.06);
  cutCap.rotation.z = -0.25;
  group.add(cutCap);

  // Dark floral tip nub at bottom (u=0)
  const tipGeo = trackGeo(reg, new THREE.SphereGeometry(0.048, 10, 10));
  const tipMat = trackMat(
    reg,
    new THREE.MeshStandardMaterial({ color: 0x291d12, roughness: 0.92 }),
  );
  const tipMesh = new THREE.Mesh(tipGeo, tipMat);
  tipMesh.position.set(-0.8, -0.46, 0);
  group.add(tipMesh);

  return group;
}

/**
 * 3. Navel Orange (5 distinct parts: dimpled oblate citrus Lathe body, recessed navel dimple,
 *    green calyx button, 6 star sepals, and glossy citrus leaf).
 */
export function createOrangeModel(reg: DisposableRegistry): THREE.Group {
  const group = new THREE.Group();
  group.name = 'fruit_orange';

  const { albedo, bump } = createPeelTextures(reg, {
    seed: 303,
    baseTop: '#FB923C',
    baseMid: '#F97316',
    baseBottom: '#EA580C',
    speckles: { color: '#C2410C', count: 380, maxRadius: 0.95, alpha: 0.35 },
    bumpStrength: 1.0,
  });

  const profile: THREE.Vector2[] = [];
  const steps = 26;
  for (let i = 0; i <= steps; i++) {
    const t = i / steps;
    const theta = t * Math.PI;
    const r = Math.sin(theta) * 0.78;
    let y = -Math.cos(theta) * 0.72;
    if (t > 0.9) y -= ((t - 0.9) / 0.1) * 0.05;
    if (t < 0.1) y += ((0.1 - t) / 0.1) * 0.05;
    profile.push(new THREE.Vector2(Math.max(0.001, r), y));
  }
  const rindGeo = trackGeo(reg, new THREE.LatheGeometry(profile, 42));
  const rindMat = trackMat(
    reg,
    new THREE.MeshPhysicalMaterial({
      map: albedo,
      bumpMap: bump,
      bumpScale: 0.018,
      roughness: 0.36,
      metalness: 0.0,
      clearcoat: 0.38,
      clearcoatRoughness: 0.34,
    }),
  );
  group.add(new THREE.Mesh(rindGeo, rindMat));

  // Green button calyx + 6 star sepals
  const calyxMat = trackMat(
    reg,
    new THREE.MeshStandardMaterial({ color: 0x166534, roughness: 0.55 }),
  );
  const buttonGeo = trackGeo(reg, new THREE.CylinderGeometry(0.06, 0.08, 0.05, 12));
  const button = new THREE.Mesh(buttonGeo, calyxMat);
  button.position.set(0, 0.68, 0);
  group.add(button);

  const sepalGeo = trackGeo(reg, new THREE.ConeGeometry(0.032, 0.12, 4));
  for (let i = 0; i < 6; i++) {
    const a = (i / 6) * Math.PI * 2;
    const sepal = new THREE.Mesh(sepalGeo, calyxMat);
    sepal.position.set(Math.cos(a) * 0.09, 0.68, Math.sin(a) * 0.09);
    sepal.rotation.set(Math.PI * 0.48, -a, 0, 'YXZ');
    group.add(sepal);
  }

  const leafMat = trackMat(
    reg,
    new THREE.MeshPhysicalMaterial({
      color: 0x15803d,
      roughness: 0.3,
      clearcoat: 0.4,
      side: THREE.DoubleSide,
    }),
  );
  const veinMat = trackMat(
    reg,
    new THREE.MeshStandardMaterial({ color: 0x86efac, roughness: 0.5 }),
  );
  const leaf = createBotanicalLeaf(reg, 0.46, 0.2, leafMat, veinMat);
  leaf.position.set(0.03, 0.7, 0.02);
  leaf.rotation.set(0.2, -0.8, -0.45);
  group.add(leaf);

  return group;
}

/**
 * 4. Garden Strawberry (4 distinct parts: conical heart-shaped Lathe receptacle,
 *    InstancedMesh of 76 phyllotaxis golden achenes/seeds, 7-sepal leafy hull crown, and stem).
 */
export function createStrawberryModel(reg: DisposableRegistry): THREE.Group {
  const group = new THREE.Group();
  group.name = 'fruit_strawberry';

  const { albedo, bump } = createPeelTextures(reg, {
    seed: 404,
    baseTop: '#F43F5E',
    baseMid: '#E11D48',
    baseBottom: '#9F1239',
    bumpStrength: 0.6,
  });

  const radiusAtT = (t: number): number => {
    // t=0 pointed tip at bottom, t=1 rounded shoulder at top
    const shoulder = Math.sin(Math.pow(t, 0.65) * Math.PI);
    return Math.max(0.002, shoulder * (0.28 + 0.48 * Math.pow(t, 0.55)));
  };
  const yAtT = (t: number): number => -0.68 + t * 1.26;

  const profile: THREE.Vector2[] = [];
  const steps = 28;
  for (let i = 0; i <= steps; i++) {
    const t = i / steps;
    profile.push(new THREE.Vector2(radiusAtT(t), yAtT(t)));
  }

  const berryGeo = trackGeo(reg, new THREE.LatheGeometry(profile, 38));
  const berryMat = trackMat(
    reg,
    new THREE.MeshPhysicalMaterial({
      map: albedo,
      bumpMap: bump,
      bumpScale: 0.012,
      roughness: 0.28,
      clearcoat: 0.52,
      clearcoatRoughness: 0.25,
    }),
  );
  group.add(new THREE.Mesh(berryGeo, berryMat));

  // InstancedMesh of 76 golden achenes arranged in Fibonacci phyllotaxis
  const seedCount = 76;
  const seedGeo = trackGeo(reg, new THREE.SphereGeometry(0.022, 6, 6));
  seedGeo.scale(0.75, 1.35, 0.6);
  const seedMat = trackMat(
    reg,
    new THREE.MeshStandardMaterial({ color: 0xfde047, roughness: 0.45, metalness: 0.05 }),
  );
  const seedsInstanced = new THREE.InstancedMesh(seedGeo, seedMat, seedCount);
  const dummy = new THREE.Object3D();
  const goldenAngle = Math.PI * (3 - Math.sqrt(5));
  for (let i = 0; i < seedCount; i++) {
    const t = 0.09 + (i / seedCount) * 0.8;
    const r = radiusAtT(t) * 1.005;
    const y = yAtT(t);
    const phi = i * goldenAngle;
    dummy.position.set(Math.cos(phi) * r, y, Math.sin(phi) * r);
    dummy.lookAt(0, y + 0.18, 0);
    dummy.updateMatrix();
    seedsInstanced.setMatrixAt(i, dummy.matrix);
  }
  seedsInstanced.instanceMatrix.needsUpdate = true;
  group.add(seedsInstanced);

  // Leafy green calyx crown (7 curved sepals) + pedicel stem
  const hullMat = trackMat(
    reg,
    new THREE.MeshStandardMaterial({
      color: 0x16a34a,
      roughness: 0.48,
      side: THREE.DoubleSide,
    }),
  );
  const veinMat = trackMat(
    reg,
    new THREE.MeshStandardMaterial({ color: 0x4ade80, roughness: 0.55 }),
  );
  for (let i = 0; i < 7; i++) {
    const a = (i / 7) * Math.PI * 2;
    const sepal = createBotanicalLeaf(reg, 0.34, 0.13, hullMat, veinMat);
    sepal.position.set(Math.cos(a) * 0.08, 0.56, Math.sin(a) * 0.08);
    sepal.rotation.set(0.35, -a + Math.PI * 0.5, 0, 'YXZ');
    group.add(sepal);
  }

  const stemCurve = new THREE.CatmullRomCurve3([
    new THREE.Vector3(0, 0.54, 0),
    new THREE.Vector3(0.02, 0.72, 0.01),
    new THREE.Vector3(0.07, 0.86, 0.03),
  ]);
  const stemGeo = trackGeo(reg, new THREE.TubeGeometry(stemCurve, 10, 0.026, 7, false));
  group.add(new THREE.Mesh(stemGeo, hullMat));

  return group;
}

/**
 * 5. Concord Grape Cluster (4 distinct parts: woody main rachis stem, InstancedMesh of
 *    24 velvety epicuticular-bloom berries in a conical bunch, coiled vine tendril, and vine leaf).
 */
export function createGrapeClusterModel(reg: DisposableRegistry): THREE.Group {
  const group = new THREE.Group();
  group.name = 'fruit_grape';

  // Woody central rachis
  const rachisCurve = new THREE.CatmullRomCurve3([
    new THREE.Vector3(0, -0.58, 0),
    new THREE.Vector3(0.02, 0.15, 0.02),
    new THREE.Vector3(-0.04, 0.76, -0.02),
    new THREE.Vector3(0.08, 0.96, 0.04),
  ]);
  const rachisGeo = trackGeo(reg, new THREE.TubeGeometry(rachisCurve, 18, 0.032, 7, false));
  const stemMat = trackMat(
    reg,
    new THREE.MeshStandardMaterial({ color: 0x5c4028, roughness: 0.82 }),
  );
  group.add(new THREE.Mesh(rachisGeo, stemMat));

  // Coiled helical tendril near top stem
  const tendrilPts: THREE.Vector3[] = [];
  for (let i = 0; i <= 26; i++) {
    const t = i / 26;
    const angle = t * Math.PI * 5.5;
    const rad = 0.08 * (1 - t * 0.55);
    tendrilPts.push(
      new THREE.Vector3(
        0.06 + t * 0.36 + Math.cos(angle) * rad,
        0.72 + Math.sin(angle) * rad,
        Math.sin(angle * 0.8) * rad,
      ),
    );
  }
  const tendrilGeo = trackGeo(
    reg,
    new THREE.TubeGeometry(new THREE.CatmullRomCurve3(tendrilPts), 32, 0.012, 6, false),
  );
  const tendrilMat = trackMat(
    reg,
    new THREE.MeshStandardMaterial({ color: 0x4d7c0f, roughness: 0.6 }),
  );
  group.add(new THREE.Mesh(tendrilGeo, tendrilMat));

  // InstancedMesh of 24 plump Concord berries with epicuticular wax sheen
  const berryCoords: Array<[number, number, number, number]> = [];
  const tiers = [
    { y: 0.44, count: 6, radius: 0.31, size: 0.19 },
    { y: 0.19, count: 7, radius: 0.35, size: 0.195 },
    { y: -0.07, count: 5, radius: 0.27, size: 0.185 },
    { y: -0.32, count: 4, radius: 0.19, size: 0.175 },
    { y: -0.54, count: 2, radius: 0.09, size: 0.165 },
  ];
  tiers.forEach((tier, tIdx) => {
    for (let i = 0; i < tier.count; i++) {
      const a = (i / tier.count) * Math.PI * 2 + tIdx * 0.55;
      berryCoords.push([
        Math.cos(a) * tier.radius,
        tier.y + ((i % 2) - 0.5) * 0.04,
        Math.sin(a) * tier.radius,
        tier.size,
      ]);
    }
  });

  const grapeGeo = trackGeo(reg, new THREE.SphereGeometry(1, 20, 20));
  grapeGeo.scale(0.96, 1.08, 0.96);
  const grapeMat = trackMat(
    reg,
    new THREE.MeshPhysicalMaterial({
      color: 0x4c1d95,
      roughness: 0.34,
      metalness: 0.02,
      sheen: 0.72,
      sheenRoughness: 0.45,
      sheenColor: new THREE.Color(0xc4b5fd),
      clearcoat: 0.25,
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
    colorVar.setHSL(0.76 + (idx % 4) * 0.012, 0.62, 0.24 + (idx % 3) * 0.035);
    berries.setColorAt(idx, colorVar);
  });
  berries.instanceMatrix.needsUpdate = true;
  if (berries.instanceColor) berries.instanceColor.needsUpdate = true;
  group.add(berries);

  const leafMat = trackMat(
    reg,
    new THREE.MeshStandardMaterial({
      color: 0x15803d,
      roughness: 0.46,
      side: THREE.DoubleSide,
    }),
  );
  const leaf = createBotanicalLeaf(reg, 0.44, 0.24, leafMat, tendrilMat);
  leaf.position.set(-0.04, 0.72, 0.02);
  leaf.rotation.set(0.25, 2.2, -0.5);
  group.add(leaf);

  return group;
}

/**
 * 6. Ruby Pomegranate (4 distinct parts: subtly 6-ribbed leathery Lathe body,
 *    tubular calyx neck, 6-pointed flared royal crown lobes, and golden inner stamens).
 */
export function createPomegranateModel(reg: DisposableRegistry): THREE.Group {
  const group = new THREE.Group();
  group.name = 'fruit_pomegranate';

  const { albedo, bump } = createPeelTextures(reg, {
    seed: 606,
    baseTop: '#9F1239',
    baseMid: '#BE123C',
    baseBottom: '#881337',
    speckles: { color: '#FBBF24', count: 190, maxRadius: 1.3, alpha: 0.22 },
    bumpStrength: 0.7,
  });

  const profile: THREE.Vector2[] = [];
  const steps = 26;
  for (let i = 0; i <= steps; i++) {
    const t = i / steps;
    const theta = t * Math.PI;
    const r = Math.sin(theta) * 0.77;
    const y = -Math.cos(theta) * 0.73;
    profile.push(new THREE.Vector2(Math.max(0.002, r), y));
  }
  const bodyGeo = trackGeo(reg, new THREE.LatheGeometry(profile, 42));
  // Characteristic subtle 6-sided pomegranate locule ribbing
  const pos = bodyGeo.attributes.position;
  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i);
    const y = pos.getY(i);
    const z = pos.getZ(i);
    const angle = Math.atan2(z, x);
    const rib = 1 + 0.032 * Math.cos(angle * 6);
    pos.setXYZ(i, x * rib, y, z * rib);
  }
  bodyGeo.computeVertexNormals();

  const rindMat = trackMat(
    reg,
    new THREE.MeshPhysicalMaterial({
      map: albedo,
      bumpMap: bump,
      bumpScale: 0.014,
      roughness: 0.42,
      clearcoat: 0.22,
      clearcoatRoughness: 0.45,
    }),
  );
  group.add(new THREE.Mesh(bodyGeo, rindMat));

  // Tubular crown collar
  const neckGeo = trackGeo(reg, new THREE.CylinderGeometry(0.16, 0.13, 0.16, 12, 1, true));
  const neckMesh = new THREE.Mesh(neckGeo, rindMat);
  neckMesh.position.set(0, 0.75, 0);
  group.add(neckMesh);

  // 6 flared triangular crown lobes
  const lobeGeo = trackGeo(reg, new THREE.ConeGeometry(0.055, 0.18, 4));
  for (let i = 0; i < 6; i++) {
    const a = (i / 6) * Math.PI * 2;
    const lobe = new THREE.Mesh(lobeGeo, rindMat);
    lobe.position.set(Math.cos(a) * 0.15, 0.86, Math.sin(a) * 0.15);
    lobe.rotation.set(-0.38, -a + Math.PI * 0.5, 0, 'YXZ');
    group.add(lobe);
  }

  // Inner dried golden stamens
  const stamenGeo = trackGeo(reg, new THREE.CylinderGeometry(0.008, 0.008, 0.12, 5));
  const stamenMat = trackMat(
    reg,
    new THREE.MeshStandardMaterial({ color: 0xd97706, roughness: 0.75 }),
  );
  for (let i = 0; i < 8; i++) {
    const a = (i / 8) * Math.PI * 2;
    const st = new THREE.Mesh(stamenGeo, stamenMat);
    st.position.set(Math.cos(a) * 0.06, 0.78, Math.sin(a) * 0.06);
    group.add(st);
  }

  return group;
}

/**
 * 7. Tropical Guava (4 distinct parts: pyriform pear-ovoid Lathe body, curved woody stem,
 *    pair of opposite elliptical leaves, and blossom-end remnant ring).
 */
export function createGuavaModel(reg: DisposableRegistry): THREE.Group {
  const group = new THREE.Group();
  group.name = 'fruit_guava';

  const { albedo, bump } = createPeelTextures(reg, {
    seed: 707,
    baseTop: '#65A30D',
    baseMid: '#84CC16',
    baseBottom: '#4D7C0F',
    speckles: { color: '#ECFCCB', count: 180, maxRadius: 1.1, alpha: 0.3 },
    bumpStrength: 0.65,
  });

  // Pyriform (pear-ovoid) silhouette
  const profile: THREE.Vector2[] = [];
  const steps = 28;
  for (let i = 0; i <= steps; i++) {
    const t = i / steps;
    const theta = t * Math.PI;
    // Wider lower belly (t ~ 0.38), narrower upper neck (t ~ 0.78)
    const pyriform = 1.08 - 0.26 * t;
    const r = Math.sin(theta) * 0.72 * pyriform;
    const y = -Math.cos(theta) * 0.82;
    profile.push(new THREE.Vector2(Math.max(0.002, r), y));
  }
  const bodyGeo = trackGeo(reg, new THREE.LatheGeometry(profile, 38));
  const skinMat = trackMat(
    reg,
    new THREE.MeshPhysicalMaterial({
      map: albedo,
      bumpMap: bump,
      bumpScale: 0.013,
      roughness: 0.38,
      clearcoat: 0.25,
      clearcoatRoughness: 0.4,
    }),
  );
  group.add(new THREE.Mesh(bodyGeo, skinMat));

  const woodyMat = trackMat(
    reg,
    new THREE.MeshStandardMaterial({ color: 0x573b26, roughness: 0.85 }),
  );
  const stemCurve = new THREE.CatmullRomCurve3([
    new THREE.Vector3(0, 0.78, 0),
    new THREE.Vector3(-0.04, 0.96, 0.02),
  ]);
  group.add(new THREE.Mesh(trackGeo(reg, new THREE.TubeGeometry(stemCurve, 8, 0.026, 7, false)), woodyMat));

  const leafMat = trackMat(
    reg,
    new THREE.MeshStandardMaterial({
      color: 0x166534,
      roughness: 0.44,
      side: THREE.DoubleSide,
    }),
  );
  const veinMat = trackMat(
    reg,
    new THREE.MeshStandardMaterial({ color: 0xa3e635, roughness: 0.5 }),
  );
  const leafA = createBotanicalLeaf(reg, 0.42, 0.19, leafMat, veinMat);
  leafA.position.set(-0.02, 0.84, 0.01);
  leafA.rotation.set(0.2, 0.4, -0.55);
  group.add(leafA);

  // Basal blossom-end calyx ring
  const ringGeo = trackGeo(reg, new THREE.TorusGeometry(0.075, 0.018, 8, 16));
  const ring = new THREE.Mesh(ringGeo, woodyMat);
  ring.position.set(0, -0.81, 0);
  ring.rotation.x = Math.PI * 0.5;
  group.add(ring);

  return group;
}

/**
 * Precision Optical Calibrator Gimbal (4 sub-parts: outer dark anodized housing ring,
 * inner counter-rotating calibration ring, 4 cardinal fiducial brackets, and restrained emerald reticle).
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
      color: 0x16211b,
      roughness: 0.32,
      metalness: 0.82,
    }),
  );

  const outerGeo = trackGeo(reg, new THREE.TorusGeometry(1.62, 0.018, 12, 96));
  const outerRing = new THREE.Mesh(outerGeo, anodizedMat);
  group.add(outerRing);

  const reticleMat = trackMat(
    reg,
    new THREE.MeshBasicMaterial({
      color: 0x10b981,
      transparent: true,
      opacity: 0.34,
    }),
  );
  const innerGeo = trackGeo(reg, new THREE.TorusGeometry(1.46, 0.007, 8, 96));
  const innerRing = new THREE.Mesh(innerGeo, reticleMat);
  group.add(innerRing);

  const tickGeo = trackGeo(reg, new THREE.BoxGeometry(0.09, 0.016, 0.024));
  for (let i = 0; i < 4; i++) {
    const a = (i / 4) * Math.PI * 2;
    const tick = new THREE.Mesh(tickGeo, reticleMat);
    tick.position.set(Math.cos(a) * 1.54, Math.sin(a) * 1.54, 0);
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
      homePos: new THREE.Vector3(2.35, 0.32, 0.35),
      scrollDrift: new THREE.Vector3(-0.7, 1.35, -1.2),
      rotSpeed: new THREE.Vector3(0.14, 0.34, 0.08),
      floatPhase: 0.0,
      baseScale: 1.06,
    },
    {
      id: 'orange',
      label: 'Orange',
      group: createOrangeModel(reg),
      homePos: new THREE.Vector3(-3.35, 1.55, -1.6),
      scrollDrift: new THREE.Vector3(1.15, -2.1, 1.65),
      rotSpeed: new THREE.Vector3(0.18, -0.28, 0.11),
      floatPhase: 1.1,
      baseScale: 0.94,
    },
    {
      id: 'banana',
      label: 'Banana',
      group: createBananaModel(reg),
      homePos: new THREE.Vector3(4.15, -1.65, -1.85),
      scrollDrift: new THREE.Vector3(-1.45, 2.6, 1.35),
      rotSpeed: new THREE.Vector3(0.12, 0.26, -0.15),
      floatPhase: 2.2,
      baseScale: 1.02,
    },
    {
      id: 'strawberry',
      label: 'Strawberry',
      group: createStrawberryModel(reg),
      homePos: new THREE.Vector3(0.68, -1.78, 0.65),
      scrollDrift: new THREE.Vector3(-2.1, 2.4, -0.9),
      rotSpeed: new THREE.Vector3(-0.16, 0.38, 0.12),
      floatPhase: 3.1,
      baseScale: 0.88,
    },
    {
      id: 'grape',
      label: 'Grape',
      group: createGrapeClusterModel(reg),
      homePos: new THREE.Vector3(-2.85, -1.75, -1.25),
      scrollDrift: new THREE.Vector3(1.5, 2.85, 0.95),
      rotSpeed: new THREE.Vector3(0.11, 0.25, 0.09),
      floatPhase: 4.0,
      baseScale: 0.96,
    },
    {
      id: 'pomegranate',
      label: 'Pomegranate',
      group: createPomegranateModel(reg),
      homePos: new THREE.Vector3(3.85, 2.05, -2.4),
      scrollDrift: new THREE.Vector3(-1.6, -2.75, 1.8),
      rotSpeed: new THREE.Vector3(-0.14, 0.29, -0.08),
      floatPhase: 4.9,
      baseScale: 0.95,
    },
    {
      id: 'guava',
      label: 'Guava',
      group: createGuavaModel(reg),
      homePos: new THREE.Vector3(-1.15, 2.35, -2.75),
      scrollDrift: new THREE.Vector3(2.3, -2.5, 1.55),
      rotSpeed: new THREE.Vector3(0.15, -0.31, 0.1),
      floatPhase: 5.7,
      baseScale: 0.92,
    },
  ];
}
