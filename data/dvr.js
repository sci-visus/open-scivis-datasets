/*
dvr.js - v2.3 - public domain - Pavol Klacansky

Surface and volume renderer using WebGL 2.

Example HTML to use the demo viewer. It reads a raw data file called 'skull_256x256x256_uint8.raw'.

```
<script type="module">
import * as dvr from './dvr.js'

let renderer

function
loadDataset(url, width, height, depth, boxWidth, boxHeight, boxDepth)
{
	const viewer = document.querySelector('#viewer')
	const progress = viewer.querySelector('#progress')

	viewer.style.display = 'block'
	progress.style.display = 'inline-block'
	viewer.querySelector('canvas').style.display = 'none'

	// set to 0 % in case of slow connection
	progress.textContent = `Downloading preview (0 %)`

	if (!renderer) {
		renderer = new dvr.ExampleViewer(viewer.querySelector('canvas'))
	}

	fetch(url)
		.then(response => {
			if (!response.ok)
				throw new Error('Response not ok: ' + response.statusText)

			const totalBytes = response.headers.get('Content-Length')
			const array = new Uint8Array(totalBytes)

			const reader = response.body.getReader()
			let receivedBytes = 0

			// stream the data into an array in chunks to allow for progress bar
			return reader.read().then(function processChunk(result) {
				if (result.done) {
					progress.textContent = `Downloading preview (100 %)`
					return array.buffer
				}

				array.set(result.value, receivedBytes)
				receivedBytes += result.value.length

				const percent = Math.round(100*receivedBytes/totalBytes)
				progress.textContent = `Downloading preview (${percent} %)`

				return reader.read().then(processChunk)
			})
		}).then(data => {
			const iso_select = document.querySelector('#isovalue_select')
			iso_select.value = 0.5

			renderer.uploadData(new Uint8Array(data), width, height, depth, boxWidth, boxHeight, boxDepth)
			renderer.isovalue(0.5)
			progress.style.display = 'none'
			viewer.querySelector('canvas').style.display = 'inline-block'
		})
}


function
init()
{
	loadDataset('skull_256x256x256_uint8.raw', 256, 256, 256, 1, 1, 1)
	document.querySelector('#isovalue_select').addEventListener('input', e => renderer.isovalue(e.target.value))
}

document.addEventListener('DOMContentLoaded', init)
</script>


<style>
#viewer {
	display: none;
	text-align: center;
}

#viewer #progress, #viewer canvas {
	display: inline-block;
	width: 800px;
}

#viewer #progress {
	height: 800px;
	vertical-align: middle;
	line-height: 800px;
}

#webgl2_error {
	display: none;
}
</style>


<div id="viewer">
	<div id="progress"></div>
	<canvas width="800" height="800"></canvas>
	<p id="webgl2_error">
		WebGL 2 is not supported; try recent Firefox or Chrome.
	<div>
		<label>Isovalue:<input id="isovalue_select" type="range" min="0" max="1" step="0.01"></label>
	</div>
</div>
```


Potential todos:

- show axes
- temporal noise component

Versions:

2.3 - generalize boundingBox to box function to allow different sized boxes
2.2 - fix precision specifier in fragment shaders
2.1 - fix missing volume texture binding during slice rendering
2.0 - fix sRGB on early return in ray cast; support transparent isosurface; add blending; support axis aligned slice rendering; switch to immediate mode rendering API; provide example viewer
1.1 - replace reduce/map with loops to improve performance (roughly 10x in Chrome)
1.0 - handle lose/restore WebGL context
0.9 - remove WebVR (only supported by Firefox); support float64 input
0.8 - rename from pk-dvr.js to dvr.js
0.7 - fix sRGB output for volume rendering
0.6 - maximum number of iterations for ray casting to avoid GPU hangs
0.5 - switch to async initialization due to VR setup being async
0.4 - render bounding box
0.3 - fix virtual reality rendering
0.2 - add noise function
0.1 - initial release supporting volume and isosurface rendering
*/


export {ExampleViewer, Renderer}


const positions = new Float32Array([
	0.0, 0.0, 0.0,
	1.0, 0.0, 0.0,
	0.0, 1.0, 0.0,
	1.0, 1.0, 0.0,
	0.0, 0.0, 1.0,
	1.0, 0.0, 1.0,
	0.0, 1.0, 1.0,
	1.0, 1.0, 1.0,
])
// some GPUs have lower performance if uint8 is used
const indices = new Uint16Array([
	// bottom
	0, 2, 1,
	2, 3, 1,
	// top
	4, 5, 6,
	6, 5, 7,
	// left
	2, 0, 6,
	6, 0, 4,
	// right
	1, 3, 5,
	5, 3, 7,
	// back
	3, 2, 7,
	7, 2, 6,
	// front
	0, 1, 4,
	4, 1, 5,
])
// some GPUs have lower performance if uint8 is used
const boxIndices = new Uint16Array([
	0, 1,
	0, 2,
	1, 3,
	2, 3,
	0, 4,
	1, 5,
	2, 6,
	3, 7,
	4, 5,
	4, 6,
	5, 7,
	6, 7,
])




const vertSrcBox = `#version 300 es

layout (location = 0) in vec4 pos;
layout (std140) uniform Matrices {
	mat4 model;
	mat4 view;
	mat4 projection;
} matrices;

void
main(void)
{
	gl_Position = matrices.projection*matrices.view*matrices.model*pos;
}
`.trim()


const fragSrcBox = `
#version 300 es

precision highp float;

layout(location = 0) out vec4 color;

void
main(void)
{
	color = vec4(0.0, 0.0, 0.0, 1.0);
}
`.trim()




function
mkVertSrc(renderingMode)
{
	return `
#version 300 es

layout(location = 0) in vec4 pos;
layout(std140) uniform Matrices {
	mat4 model;
	mat4 view;
	mat4 projection;
} matrices;

out vec3 v_pos;
out vec3 v_world_pos;
out vec3 eye;
out mat4 to_world;
out mat3 to_worldn;
out float zScaling;

void
main(void)
{
	// needed to scale tFar in fragment shader, because we can't use infinite far plane projection matrix
	zScaling = matrices.projection[2][2];

	// TODO: precompute inverse
	to_world = matrices.view*matrices.model;
	mat4 inv = inverse(to_world);
	to_worldn = transpose(mat3(inv));
	eye = (inv*vec4(0.0, 0.0, 0.0, 1.0)).xyz;
	vec4 position = pos;
	v_pos = position.xyz;
	v_world_pos = (matrices.view*matrices.model*position).xyz;
	gl_Position = matrices.projection*matrices.view*matrices.model*position;
}
`.trim()
}




function
mkFragSrc(renderingMode)
{
	return `
#version 300 es

#define SURFACE 0
#define VOLUME 1
#define METHOD ${(renderingMode === 'surface') ? 'SURFACE' : 'VOLUME'}

precision highp float;

uniform highp sampler3D volume_sampler;
uniform highp sampler2D depth_sampler;
uniform highp sampler2D transfer_function_sampler;

uniform float isovalue;
uniform vec4 surface_color;

in vec3 v_pos;
in vec3 v_world_pos;
in vec3 eye;
in mat4 to_world;
in mat3 to_worldn;
in float zScaling;

layout(location = 0) out vec4 color;


// from internet (unknown original source)
float
rand(vec2 co)
{
	return fract(sin(dot(co.xy, vec2(12.9898, 78.233)))*43758.5453);
}


// central difference
vec3
gradient(in highp sampler3D s, vec3 p, float dt)
{
	vec2 e = vec2(dt, 0.0);

	return vec3(texture(s, p - e.xyy).r - texture(s, p + e.xyy).r,
		texture(s, p - e.yxy).r - texture(s, p + e.yxy).r,
		texture(s, p - e.yyx).r - texture(s, p + e.yyx).r);
}


float
linear_to_srgb(float linear)
{
	if (linear <= 0.0031308) {
		return 12.92*linear;
	} else {
		return (1.0 + 0.055)*pow(linear, 1.0/2.4) - 0.055;
	}
}


void
main(void)
{
	const vec3 light_pos = vec3(1.0, 1.0, 1.0);
	vec3 o = eye;
	vec3 d = normalize(v_pos - o);

	// intersect aabb
	vec3 near = min(-o/d, (vec3(1.0) - o)/d);
	vec3 far = max(-o/d, (vec3(1.0) - o)/d);
	float tnear = max(near.x, max(near.y, near.z));
	float tfar  = min(far.x, min(far.y, far.z));
	if (tnear > tfar) {
		discard;
	}

	// stop at geometry if there is any (do ratio of z coordinate depth and z coordinate of current fragment)
	float depth = texelFetch(depth_sampler, ivec2(gl_FragCoord.xy), 0).r;
	tfar *= min((zScaling + gl_FragCoord.z)/(zScaling + depth), 1.0);

	ivec3 size = textureSize(volume_sampler, 0);
	int max_size = max(size.x, max(size.y, size.z));

	// compute step size (3D DDA)
	vec3 cell_size = 1.0/vec3(size);
	vec3 dts = cell_size/abs(d);
	float dt = min(dts.x, min(dts.y, dts.z));

	color = vec4(0.0, 0.0, 0.0, 0.0);
	float prev_value = 0.0;
	float t = tnear + dt*rand(gl_FragCoord.xy);

	for (int i = 0; i < max_size; ++i) {
		if (t >= tfar) {
			break;
		}

		vec3 p = o + t*d;
		float value = texture(volume_sampler, p).r;
#if (METHOD == SURFACE)
		if (sign(value - isovalue) != sign(prev_value - isovalue)) {
			// linear approximation of intersection point
			vec3 prev_p = p - dt*d;
			float a = (isovalue - prev_value)/(value - prev_value);
			vec3 inter_p = (1.0 - a)*(p - dt*d) + a*p;
			// TODO: sample at different dt for each axis to avoid having undo scaling
			vec3 nn = gradient(volume_sampler, inter_p, dt);

			// TODO: can we optimize somehow?
			vec3 world_p = (to_world*vec4(inter_p, 1.0)).xyz;
			vec3 n = normalize(to_worldn*nn);
			vec3 light_dir = normalize(light_pos - world_p);
			vec3 h = normalize(light_dir - world_p); // eye is at origin
			const float ambient = 0.2;
			float diffuse = 0.6*clamp(dot(light_dir, n), 0.0, 1.0);
			float specular = 0.2*pow(clamp(dot(h, n), 0.0, 1.0), 100.0);
			float distance = length(world_p); // eye is at origin
			color.rgb += (1.0 - color.a)*surface_color.a*surface_color.rgb*(ambient + (diffuse + specular)/distance);
			color.a += (1.0 - color.a)*surface_color.a;
			if (color.a == 1.0) {
				color.rgb = vec3(linear_to_srgb(color.r), linear_to_srgb(color.g), linear_to_srgb(color.b));
				return;
			}
		}
		prev_value = value;
#elif (METHOD == VOLUME)
		if (color.a > 0.95) {
			color.rgb = vec3(linear_to_srgb(color.r), linear_to_srgb(color.g), linear_to_srgb(color.b));
			return;
		}
		vec4 sample_color = texture(transfer_function_sampler, vec2(value, 0.0));
		color.rgb += (1.0 - color.a)*sample_color.a*sample_color.rgb;
		color.a += (1.0 - color.a)*sample_color.a;
#endif
		t += dt;
	}

	color.rgb = vec3(linear_to_srgb(color.r), linear_to_srgb(color.g), linear_to_srgb(color.b));
}
`.trim()
}


const vertSrcSlice = `
#version 300 es

layout(location = 0) in vec4 pos;
layout(std140) uniform Matrices {
	mat4 model;
	mat4 view;
	mat4 projection;
} matrices;

out vec3 v_pos;

void
main(void)
{
	v_pos = pos.xyz;
	gl_Position = matrices.projection*matrices.view*matrices.model*pos;
}
`.trim()

const fragSrcSlice = `
#version 300 es

precision highp float;

uniform highp sampler3D volume_sampler;

in vec3 v_pos;

layout(location = 0) out vec4 color;


void
main(void)
{
	float value = texture(volume_sampler, v_pos).r;
	color = vec4(value, value, value, 1.0);
}
`.trim()



// from stackoverflow
const to_half = (() => {
	const floatView = new Float32Array(1)
	const int32View = new Int32Array(floatView.buffer)

	// This method is faster than the OpenEXR implementation (very often used, eg. in Ogre), with the additional benefit of rounding, inspired by James Tursa's half-precision code.
	return val => {
		floatView[0] = val
		const x = int32View[0]

		let bits = (x >> 16) & 0x8000 // Get the sign
		let m = (x >> 12) & 0x07ff // Keep one extra bit for rounding
		const e = (x >> 23) & 0xff // Using int is faster here

		// If zero, or denormal, or exponent underflows too much for a denormal half, return signed zero.
		if (e < 103) {
			return bits
		}

		// If NaN, return NaN. If Inf or exponent overflow, return Inf.
		if (e > 142) {
			bits |= 0x7c00
			// If exponent was 0xff and one mantissa bit was set, it means NaN, not Inf, so make sure we set one mantissa bit too.
			bits |= ((e == 255) ? 0 : 1) && (x & 0x007fffff)
			return bits
		}

		// If exponent underflows but not too much, return a denormal
		if (e < 113) {
			m |= 0x0800
			// Extra rounding may overflow and set mantissa to 0 and exponent to 1, which is OK.
			bits |= (m >> (114 - e)) + ((m >> (113 - e)) & 1)
			return bits
		}

		bits |= ((e - 112) << 10) | (m >> 1)
		// Extra rounding. An overflow will set mantissa to 0 and increment the exponent, which is OK.
		bits += m & 1
		return bits
	}
})()


function
createProgram(gl, vertSrc, fragSrc)
{
	const vertexShader = gl.createShader(gl.VERTEX_SHADER)
	gl.shaderSource(vertexShader, vertSrc)
	gl.compileShader(vertexShader)
	if (gl.getShaderInfoLog(vertexShader)) {
		console.log('WebGL:', gl.getShaderInfoLog(vertexShader))
	}

	const fragmentShader = gl.createShader(gl.FRAGMENT_SHADER)
	gl.shaderSource(fragmentShader, fragSrc)
	gl.compileShader(fragmentShader)
	if (gl.getShaderInfoLog(fragmentShader)) {
		console.log('WebGL:', gl.getShaderInfoLog(fragmentShader))
	}

	const program = gl.createProgram()
	gl.attachShader(program, vertexShader)
	gl.attachShader(program, fragmentShader)
	gl.linkProgram(program)
	if (gl.getProgramInfoLog(program)) {
		console.log('WebGL:', gl.getProgramInfoLog(program))
	}

	gl.deleteShader(vertexShader)
	gl.deleteShader(fragmentShader)

	return program
}


function
srgbToLinear(srgb)
{
	if (srgb <= 0.04045) {
		return srgb/12.92
	} else {
		return Math.pow((srgb + 0.055)/1.055, 2.4)
	}
}



function
Renderer(gl)
{
	this.gl = gl
		
	// necessary for linear filtering of float textures
	if (!this.gl.getExtension('OES_texture_float_linear')) {
		console.log('WebGL: no linear filtering for float textures')
	}

	this.gl.enable(this.gl.DEPTH_TEST)
	this.gl.enable(this.gl.CULL_FACE)
	this.gl.cullFace(this.gl.FRONT)
	this.gl.enable(gl.BLEND)
	this.gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA) // premultiplied alpha in shader

	this.surfaceProgram = createProgram(this.gl, mkVertSrc('surface'), mkFragSrc('surface'))
	this.gl.uniformBlockBinding(this.surfaceProgram, this.gl.getUniformBlockIndex(this.surfaceProgram, 'Matrices'), 0)

	this.volumeProgram = createProgram(this.gl, mkVertSrc('volume'), mkFragSrc('volume'))
	this.gl.uniformBlockBinding(this.volumeProgram, this.gl.getUniformBlockIndex(this.volumeProgram, 'Matrices'), 0)

	this.boxProgram = createProgram(this.gl, vertSrcBox, fragSrcBox)
	this.gl.uniformBlockBinding(this.boxProgram, this.gl.getUniformBlockIndex(this.boxProgram, 'Matrices'), 0)

	this.sliceProgram = createProgram(this.gl, vertSrcSlice, fragSrcSlice)
	this.gl.uniformBlockBinding(this.sliceProgram, this.gl.getUniformBlockIndex(this.sliceProgram, 'Matrices'), 0)
		
	this.vbo = this.gl.createBuffer()
	this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.vbo)
	this.gl.bufferData(this.gl.ARRAY_BUFFER, positions, this.gl.STATIC_DRAW)

	this.ebo = this.gl.createBuffer()
	this.gl.bindBuffer(this.gl.ELEMENT_ARRAY_BUFFER, this.ebo)
	this.gl.bufferData(this.gl.ELEMENT_ARRAY_BUFFER, indices, this.gl.STATIC_DRAW)

	// dummy vao
	const vao = this.gl.createVertexArray()
	this.gl.bindVertexArray(vao)

	// bounding box wireframe
	this.boxPositions = new Float32Array(3*12)
	this.boxVbo = this.gl.createBuffer()
	this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.boxVbo)
	this.gl.bufferData(this.gl.ARRAY_BUFFER, this.boxPositions, this.gl.DYNAMIC_DRAW)
	this.boxEbo = this.gl.createBuffer()
	this.gl.bindBuffer(this.gl.ELEMENT_ARRAY_BUFFER, this.boxEbo)
	this.gl.bufferData(this.gl.ELEMENT_ARRAY_BUFFER, boxIndices, this.gl.STATIC_DRAW)

	// slices
	this.slicePositions = new Float32Array(3*6)
	this.sliceVbo = this.gl.createBuffer()
	this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.sliceVbo)
	this.gl.bufferData(this.gl.ARRAY_BUFFER, this.slicePositions, this.gl.DYNAMIC_DRAW)

	// create framebuffers (since we need to read depth buffer)
	{
		// the drawingBufferWidth/Height could change in between the blit operations
		this.fboWidth = this.gl.drawingBufferWidth
		this.fboHeight = this.gl.drawingBufferHeight

		const colorRenderbuffer = this.gl.createRenderbuffer()
		this.gl.bindRenderbuffer(this.gl.RENDERBUFFER, colorRenderbuffer)
		this.gl.renderbufferStorage(this.gl.RENDERBUFFER, this.gl.SRGB8_ALPHA8, this.fboWidth, this.fboHeight)

		this.depthTexture = this.gl.createTexture()
		this.gl.bindTexture(this.gl.TEXTURE_2D, this.depthTexture)
		this.gl.texStorage2D(this.gl.TEXTURE_2D, 1, this.gl.DEPTH_COMPONENT32F, this.fboWidth, this.fboHeight)

		// framebuffer for rendering geometry
		this.fbo = this.gl.createFramebuffer()
		this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, this.fbo)
		this.gl.framebufferRenderbuffer(this.gl.FRAMEBUFFER, this.gl.COLOR_ATTACHMENT0, this.gl.RENDERBUFFER, colorRenderbuffer)
		this.gl.framebufferTexture2D(this.gl.FRAMEBUFFER, this.gl.DEPTH_ATTACHMENT, this.gl.TEXTURE_2D, this.depthTexture, 0)

		// framebuffer for rendering volume/surface
		this.volumeFbo = this.gl.createFramebuffer()
		this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, this.volumeFbo)
		this.gl.framebufferRenderbuffer(this.gl.FRAMEBUFFER, this.gl.COLOR_ATTACHMENT0, this.gl.RENDERBUFFER, colorRenderbuffer)
	}

	// create sampler objects for each texture
	this.volumeSampler = this.gl.createSampler()
	this.gl.samplerParameteri(this.volumeSampler, this.gl.TEXTURE_MAG_FILTER, this.gl.LINEAR)
	this.gl.samplerParameteri(this.volumeSampler, this.gl.TEXTURE_MIN_FILTER, this.gl.LINEAR)
	this.gl.samplerParameteri(this.volumeSampler, this.gl.TEXTURE_WRAP_S, this.gl.CLAMP_TO_EDGE)
	this.gl.samplerParameteri(this.volumeSampler, this.gl.TEXTURE_WRAP_T, this.gl.CLAMP_TO_EDGE)
	this.gl.samplerParameteri(this.volumeSampler, this.gl.TEXTURE_WRAP_R, this.gl.CLAMP_TO_EDGE)

	this.transferFunctionSampler = this.gl.createSampler()
	this.gl.samplerParameteri(this.transferFunctionSampler, this.gl.TEXTURE_MIN_FILTER, this.gl.LINEAR)
	this.gl.samplerParameteri(this.transferFunctionSampler, this.gl.TEXTURE_WRAP_S, this.gl.CLAMP_TO_EDGE)
	this.gl.samplerParameteri(this.transferFunctionSampler, this.gl.TEXTURE_WRAP_T, this.gl.CLAMP_TO_EDGE)

	this.depthSampler = this.gl.createSampler()
	this.gl.samplerParameteri(this.depthSampler, this.gl.TEXTURE_MAG_FILTER, this.gl.NEAREST)
	this.gl.samplerParameteri(this.depthSampler, this.gl.TEXTURE_MIN_FILTER, this.gl.NEAREST)


	this.ubo = this.gl.createBuffer()
	this.gl.bindBuffer(this.gl.UNIFORM_BUFFER, this.ubo)
	this.gl.bufferData(this.gl.UNIFORM_BUFFER, 16*4*3, this.gl.DYNAMIC_DRAW)

	this.gl.useProgram(this.surfaceProgram)
	this.gl.uniform1i(gl.getUniformLocation(this.surfaceProgram, 'volume_sampler'), 0)
	this.gl.uniform1i(gl.getUniformLocation(this.surfaceProgram, 'depth_sampler'), 1)

	this.gl.useProgram(this.volumeProgram)
	this.gl.uniform1i(this.gl.getUniformLocation(this.volumeProgram, 'volume_sampler'), 0)
	this.gl.uniform1i(this.gl.getUniformLocation(this.volumeProgram, 'depth_sampler'), 1)
	this.gl.uniform1i(this.gl.getUniformLocation(this.volumeProgram, 'transfer_function_sampler'), 2)

	this.gl.useProgram(this.sliceProgram)
	this.gl.uniform1i(this.gl.getUniformLocation(this.sliceProgram, 'volume_sampler'), 0)

	this.volumeTex = undefined
	this.transferFunctionTex = undefined
}


Renderer.prototype.clear = function(color) {
	this.gl.viewport(0, 0, this.fboWidth, this.fboHeight)

	this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, this.fbo)
	this.gl.clearBufferfv(this.gl.COLOR, this.fbo, color)
	this.gl.clearBufferfv(this.gl.DEPTH, this.fbo, [1.0])
}

// in [0,1] space
Renderer.prototype.box = function(viewMatrix, projectionMatrix, low, high) {
	this.boxPositions.set([
		low[0], low[1], low[2],
		high[0], low[1], low[2],
		low[0], high[1], low[2],
		high[0], high[1], low[2],
		low[0], low[1], high[2],
		high[0], low[1], high[2],
		low[0], high[1], high[2],
		high[0], high[1], high[2],
	])

	this.gl.bindBuffer(this.gl.UNIFORM_BUFFER, this.ubo)
	this.gl.bufferSubData(this.gl.UNIFORM_BUFFER, 16*4, viewMatrix)
	this.gl.bufferSubData(this.gl.UNIFORM_BUFFER, 32*4, projectionMatrix)
	this.gl.bindBufferBase(this.gl.UNIFORM_BUFFER, 0, this.ubo)

	this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, this.fbo)

	this.gl.useProgram(this.boxProgram)
	this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.boxVbo)
	this.gl.bufferSubData(this.gl.ARRAY_BUFFER, 0, this.boxPositions)
	this.gl.bindBuffer(this.gl.ELEMENT_ARRAY_BUFFER, this.boxEbo)
	this.gl.enableVertexAttribArray(0)
	this.gl.vertexAttribPointer(0, 3, this.gl.FLOAT, false, 0, 0)
	this.gl.drawElements(this.gl.LINES, boxIndices.length, this.gl.UNSIGNED_SHORT, 0)
}


Renderer.prototype.surface = function(viewMatrix, projectionMatrix, isovalue, color) {
	this.gl.bindBuffer(this.gl.UNIFORM_BUFFER, this.ubo)
	this.gl.bufferSubData(this.gl.UNIFORM_BUFFER, 16*4, viewMatrix)
	this.gl.bufferSubData(this.gl.UNIFORM_BUFFER, 32*4, projectionMatrix)
	this.gl.bindBufferBase(this.gl.UNIFORM_BUFFER, 0, this.ubo)
		
	// volume/surface rendering goes last as it needs to read depth from previous passes
	this.gl.disable(this.gl.DEPTH_TEST)
	this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, this.volumeFbo)

	this.gl.activeTexture(this.gl.TEXTURE0)
	this.gl.bindTexture(this.gl.TEXTURE_3D, this.volumeTex)
	this.gl.bindSampler(0, this.volumeSampler)

	this.gl.activeTexture(this.gl.TEXTURE1)
	this.gl.bindTexture(this.gl.TEXTURE_2D, this.depthTexture)
	this.gl.bindSampler(1, this.depthSampler)

	this.gl.useProgram(this.surfaceProgram)
	this.gl.uniform1f(this.gl.getUniformLocation(this.surfaceProgram, 'isovalue'), isovalue)
	const linearColor = [srgbToLinear(color[0]), srgbToLinear(color[1]), srgbToLinear(color[2]), color[3]]
	this.gl.uniform4fv(this.gl.getUniformLocation(this.surfaceProgram, 'surface_color'), linearColor)
	this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.vbo)
	this.gl.bindBuffer(this.gl.ELEMENT_ARRAY_BUFFER, this.ebo)
	this.gl.enableVertexAttribArray(0)
	this.gl.vertexAttribPointer(0, 3, this.gl.FLOAT, false, 0, 0)
	this.gl.drawElements(this.gl.TRIANGLES, indices.length, this.gl.UNSIGNED_SHORT, 0)

	this.gl.enable(this.gl.DEPTH_TEST)
}


// TODO: preferably have transfer function to take 0-1 values instead 0-255 values (since all other colors are 0-1)
Renderer.prototype.volume = function(viewMatrix, projectionMatrix, transferFunction) {
	// TODO: either check if the transfer function is of same size, or allow undefined transfer function to avoid reallocating new texture
	if (this.transferFunctionTex) {
		this.gl.deleteTexture(this.transferFunctionTex)
	}
	// WebGL does not support 1D textures directly, so we use 2D texture with height 1
	this.transferFunctionTex = this.gl.createTexture()
	this.gl.bindTexture(this.gl.TEXTURE_2D, this.transferFunctionTex)
	this.gl.texStorage2D(this.gl.TEXTURE_2D, 1, this.gl.SRGB8_ALPHA8, transferFunction.length/4, 1)
	this.gl.texSubImage2D(this.gl.TEXTURE_2D, 0, 0, 0, transferFunction.length/4, 1, this.gl.RGBA, this.gl.UNSIGNED_BYTE, transferFunction)

	this.gl.bindBuffer(this.gl.UNIFORM_BUFFER, this.ubo)
	this.gl.bufferSubData(this.gl.UNIFORM_BUFFER, 16*4, viewMatrix)
	this.gl.bufferSubData(this.gl.UNIFORM_BUFFER, 32*4, projectionMatrix)
	this.gl.bindBufferBase(this.gl.UNIFORM_BUFFER, 0, this.ubo)
		
	// volume/surface rendering goes last as it needs to read depth from previous passes
	this.gl.disable(this.gl.DEPTH_TEST) 
	this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, this.volumeFbo)

	this.gl.activeTexture(this.gl.TEXTURE0)
	this.gl.bindTexture(this.gl.TEXTURE_3D, this.volumeTex)
	this.gl.bindSampler(0, this.volumeSampler)

	this.gl.activeTexture(this.gl.TEXTURE1)
	this.gl.bindTexture(this.gl.TEXTURE_2D, this.depthTexture)
	this.gl.bindSampler(1, this.depthSampler)

	this.gl.activeTexture(this.gl.TEXTURE2)
	this.gl.bindTexture(this.gl.TEXTURE_2D, this.transferFunctionTex)
	this.gl.bindSampler(2, this.transferFunctionSampler)

	this.gl.useProgram(this.volumeProgram)
	this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.vbo)
	this.gl.bindBuffer(this.gl.ELEMENT_ARRAY_BUFFER, this.ebo)
	this.gl.enableVertexAttribArray(0)
	this.gl.vertexAttribPointer(0, 3, this.gl.FLOAT, false, 0, 0)
	this.gl.drawElements(this.gl.TRIANGLES, indices.length, this.gl.UNSIGNED_SHORT, 0)

	this.gl.enable(this.gl.DEPTH_TEST)
}


Renderer.prototype.sliceX = function(viewMatrix, projectionMatrix, x) {
	this.slicePositions.set([
		x, 0.0, 1.0,
		x, 0.0, 0.0,
		x, 1.0, 1.0,
		x, 1.0, 1.0,
		x, 0.0, 0.0,
		x, 1.0, 0.0,
	])

	this.gl.bindBuffer(this.gl.UNIFORM_BUFFER, this.ubo)
	this.gl.bufferSubData(this.gl.UNIFORM_BUFFER, 16*4, viewMatrix)
	this.gl.bufferSubData(this.gl.UNIFORM_BUFFER, 32*4, projectionMatrix)
	this.gl.bindBufferBase(this.gl.UNIFORM_BUFFER, 0, this.ubo)

	this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, this.fbo)

	this.gl.activeTexture(this.gl.TEXTURE0)
	this.gl.bindTexture(this.gl.TEXTURE_3D, this.volumeTex)
	this.gl.bindSampler(0, this.volumeSampler)

	this.gl.disable(this.gl.CULL_FACE)	
	this.gl.useProgram(this.sliceProgram)
	this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.sliceVbo)
	this.gl.bufferSubData(this.gl.ARRAY_BUFFER, 0, this.slicePositions)
	this.gl.enableVertexAttribArray(0)
	this.gl.vertexAttribPointer(0, 3, this.gl.FLOAT, false, 0, 0)
	this.gl.drawArrays(this.gl.TRIANGLES, 0, 6)
	this.gl.enable(this.gl.CULL_FACE)
}


Renderer.prototype.sliceY = function(viewMatrix, projectionMatrix, y) {
	this.slicePositions.set([
		0.0, y, 1.0,
		0.0, y, 0.0,
		1.0, y, 1.0,
		1.0, y, 1.0,
		0.0, y, 0.0,
		1.0, y, 0.0,
	])

	this.gl.bindBuffer(this.gl.UNIFORM_BUFFER, this.ubo)
	this.gl.bufferSubData(this.gl.UNIFORM_BUFFER, 16*4, viewMatrix)
	this.gl.bufferSubData(this.gl.UNIFORM_BUFFER, 32*4, projectionMatrix)
	this.gl.bindBufferBase(this.gl.UNIFORM_BUFFER, 0, this.ubo)

	this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, this.fbo)

	this.gl.activeTexture(this.gl.TEXTURE0)
	this.gl.bindTexture(this.gl.TEXTURE_3D, this.volumeTex)
	this.gl.bindSampler(0, this.volumeSampler)

	this.gl.disable(this.gl.CULL_FACE)	
	this.gl.useProgram(this.sliceProgram)
	this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.sliceVbo)
	this.gl.bufferSubData(this.gl.ARRAY_BUFFER, 0, this.slicePositions)
	this.gl.enableVertexAttribArray(0)
	this.gl.vertexAttribPointer(0, 3, this.gl.FLOAT, false, 0, 0)
	this.gl.drawArrays(this.gl.TRIANGLES, 0, 6)
	this.gl.enable(this.gl.CULL_FACE)
}


Renderer.prototype.sliceZ = function(viewMatrix, projectionMatrix, z) {
	this.slicePositions.set([
		0.0, 1.0, z,
		0.0, 0.0, z,
		1.0, 1.0, z,
		1.0, 1.0, z,
		0.0, 0.0, z,
		1.0, 0.0, z,
	])

	this.gl.bindBuffer(this.gl.UNIFORM_BUFFER, this.ubo)
	this.gl.bufferSubData(this.gl.UNIFORM_BUFFER, 16*4, viewMatrix)
	this.gl.bufferSubData(this.gl.UNIFORM_BUFFER, 32*4, projectionMatrix)
	this.gl.bindBufferBase(this.gl.UNIFORM_BUFFER, 0, this.ubo)

	this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, this.fbo)

	this.gl.activeTexture(this.gl.TEXTURE0)
	this.gl.bindTexture(this.gl.TEXTURE_3D, this.volumeTex)
	this.gl.bindSampler(0, this.volumeSampler)

	this.gl.disable(this.gl.CULL_FACE)	
	this.gl.useProgram(this.sliceProgram)
	this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.sliceVbo)
	this.gl.bufferSubData(this.gl.ARRAY_BUFFER, 0, this.slicePositions)
	this.gl.enableVertexAttribArray(0)
	this.gl.vertexAttribPointer(0, 3, this.gl.FLOAT, false, 0, 0)
	this.gl.drawArrays(this.gl.TRIANGLES, 0, 6)
	this.gl.enable(this.gl.CULL_FACE)
}

	
Renderer.prototype.blit = function(x0, y0, x1, y1) {
	this.gl.bindFramebuffer(this.gl.READ_FRAMEBUFFER, this.volumeFbo)
	this.gl.bindFramebuffer(this.gl.DRAW_FRAMEBUFFER, null)
	this.gl.blitFramebuffer(0, 0, this.fboWidth, this.fboHeight, x0, y0, x1, y1, this.gl.COLOR_BUFFER_BIT, this.gl.LINEAR)
}

	
// 8 and 16 bit integers are converted to float16
Renderer.prototype.uploadData = function(typedArray, width, height, depth, boxWidth, boxHeight, boxDepth) {
	if (this.volumeTex) {
		this.gl.deleteTexture(this.volumeTex)
	}
	this.volumeTex = this.gl.createTexture()
	this.gl.bindTexture(this.gl.TEXTURE_3D, this.volumeTex)

	const extent = [typedArray[0], typedArray[0]]
	for (let i = 0; i < typedArray.length; i++) {
		extent[0] = Math.min(typedArray[i], extent[0])
		extent[1] = Math.max(typedArray[i], extent[1])
	}

	this.gl.pixelStorei(this.gl.UNPACK_ALIGNMENT, 1)
	if (typedArray instanceof Int8Array || typedArray instanceof Int16Array || typedArray instanceof Uint8Array || typedArray instanceof Uint16Array) {
		const converted = new Uint16Array(typedArray.length)
		for (let i = 0; i < typedArray.length; i++) {
			converted[i] = to_half((typedArray[i] - extent[0])/(extent[1] - extent[0]))
		}
		this.gl.texStorage3D(this.gl.TEXTURE_3D, 1, this.gl.R16F, width, height, depth)
		this.gl.texSubImage3D(this.gl.TEXTURE_3D, 0, 0, 0, 0, width, height, depth, this.gl.RED, this.gl.HALF_FLOAT, converted)

	} else if (typedArray instanceof Float32Array) {
		const converted = new Float32Array(typedArray.length)
		for (let i = 0; i < typedArray.length; i++) {
			converted[i] = (typedArray[i] - extent[0])/(extent[1] - extent[0])
		}
		this.gl.texStorage3D(this.gl.TEXTURE_3D, 1, this.gl.R32F, width, height, depth)
		this.gl.texSubImage3D(this.gl.TEXTURE_3D, 0, 0, 0, 0, width, height, depth, this.gl.RED, this.gl.FLOAT, converted)
	} else if (typedArray instanceof Float64Array) {
		const converted = new Float32Array(typedArray.length)
		for (let i = 0; i < typedArray.length; i++) {
			converted[i] = (typedArray[i] - extent[0])/(extent[1] - extent[0])
		}
		this.gl.texStorage3D(this.gl.TEXTURE_3D, 1, this.gl.R32F, width, height, depth)
		this.gl.texSubImage3D(this.gl.TEXTURE_3D, 0, 0, 0, 0, width, height, depth, this.gl.RED, this.gl.FLOAT, converted)
	} else {
		console.log('Unsupported array type')
	}

	const max = Math.max(boxWidth, boxHeight, boxDepth)
	// scale(boxWidth/max, boxHeight/max, boxDepth/max)*translate(-0.5, -0.5, -0.5)
	const modelMatrix = new Float32Array([
		boxWidth/max, 0.0, 0.0, 0.0,
		0.0, boxHeight/max, 0.0, 0.0,
		0.0, 0.0, boxDepth/max, 0.0,
		boxWidth/max*-0.5, boxHeight/max*-0.5, boxDepth/max*-0.5, 1.0,
	])
	this.gl.bindBuffer(this.gl.UNIFORM_BUFFER, this.ubo)
	this.gl.bufferSubData(this.gl.UNIFORM_BUFFER, 0, new Float32Array(modelMatrix))
}




//////////////////////////////// Example Viewer ////////////////////////
function
ExampleViewer(canvas, renderingMode)
{
	this.arcballCircle = {
		center: {x: canvas.width/2, y: canvas.height/2},
		radius: Math.min(canvas.width/2, canvas.height/2),
	}

	let p0
	let q_down

	canvas.addEventListener('mousedown', e => {
		if (!(e.buttons & 1)) {
			return
		}

		q_down = this.q
		const rect = canvas.getBoundingClientRect()
		p0 = arcball_screenToSphere(this.arcballCircle, e.clientX - rect.left, e.clientY - rect.top)
	})
	canvas.addEventListener('mousemove', e => {
		if (!(e.buttons & 1)) {
			return
		}

		const rect = canvas.getBoundingClientRect()
		const p1 = arcball_screenToSphere(this.arcballCircle, e.clientX - rect.left, e.clientY - rect.top)
		const q_move = arcball_quat(p0, p1)

		this.q = quatMul(q_move, q_down)

		this.viewMatrix = quatToMat4(this.q)
		this.viewMatrix[14] = -this.viewDistance;

		this.render()
	})
	canvas.addEventListener('wheel', e => {
		e.preventDefault()

		this.viewDistance = Math.max(1, this.viewDistance + 0.1*Math.sign(e.deltaY))
		this.viewMatrix[14] = -this.viewDistance;

		this.render()
	})


	this.gl = canvas.getContext('webgl2', {alpha: false, antialias: false, depth: false, stencil: false})
	if (!this.gl) {
		console.log('WebGL: version 2 not available')
		alert('WebGL 2 is not available')
	}

	let context = this.gl.getExtension('WEBGL_lose_context')

	// handle context loss and restore
	canvas.addEventListener('webglcontextlost', e => {
		console.log('WebGL: lost context')
		e.preventDefault()
	})
	canvas.addEventListener('webglcontextrestored', () => {
		console.log('WebGL: restored context')
		this.renderer = new Renderer(this.gl)

		if (this.currentDataSet) {
			this.renderer.uploadData(this.currentDataSet['typedArray'], this.currentDataSet['width'], this.currentDataSet['height'], this.currentDataSet['depth'], this.currentDataSet['boxWidth'], this.currentDataSet['boxHeight'], this.currentDataSet['boxDepth'])
		}

		this.render()
	})

	this.isovalue = function(value) {
		console.assert(0 <= value && value <= 1, 'Isovalue must be in [0, 1] range.')
		this.isovalue_ = value
		this.render()
	}

	this.transferFunction = function(array) {
		this.transferFunction_ = new Uint8Array(array)
		this.render()
	}

	// setup scene
	{
		const nearPlane = 0.01
		const farPlane = 1000.0
		this.renderingMode = renderingMode
		this.q = quat(1.0, 0.0, 0.0, 0.0)
		this.viewDistance = 1.5
		this.viewMatrix = new Float32Array([
			1.0, 0.0, 0.0, 0.0,
			0.0, 1.0, 0.0, 0.0,
			0.0, 0.0, 1.0, 0.0,
			0.0, 0.0, -this.viewDistance, 1.0,
		])
		this.projectionMatrix = new Float32Array([
			1.0, 0.0, 0.0, 0.0,
			0.0, 1.0, 0.0, 0.0,
			0.0, 0.0, -(farPlane + nearPlane)/(farPlane - nearPlane), -1.0,
			0.0, 0.0, -2.0*farPlane*nearPlane/(farPlane - nearPlane), 0.0,
		])
		this.transferFunction_ = new Uint8Array([0, 0, 0, 0, 255, 255, 255, 255])
		this.isovalue_ = 0.0
	}

	this.renderer = new Renderer(this.gl)
}


ExampleViewer.prototype.render = function() {
	this.renderer.clear([255/255, 246/255, 213/255, 1.0])
	this.renderer.box(this.viewMatrix, this.projectionMatrix, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
	this.renderer.surface(this.viewMatrix, this.projectionMatrix, this.isovalue_, [0.9, 0.4, 0.4, 1.0])
	this.renderer.blit(this.gl.drawingBufferWidth/2, 0, this.gl.drawingBufferWidth, this.gl.drawingBufferHeight/2)

	this.renderer.clear([255/255, 200/255, 213/255, 1.0])
	this.renderer.box(this.viewMatrix, this.projectionMatrix, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
	this.renderer.volume(this.viewMatrix, this.projectionMatrix, this.transferFunction_)
	this.renderer.blit(0, 0, this.gl.drawingBufferWidth/2, this.gl.drawingBufferHeight/2)

	this.renderer.clear([0.25, 0.5, 1.0, 1.0])
	this.renderer.sliceX(this.viewMatrix, this.projectionMatrix, this.isovalue_)
	this.renderer.sliceY(this.viewMatrix, this.projectionMatrix, this.isovalue_)
	this.renderer.sliceZ(this.viewMatrix, this.projectionMatrix, this.isovalue_)
	this.renderer.blit(0, this.gl.drawingBufferHeight/2, this.gl.drawingBufferWidth/2, this.gl.drawingBufferHeight)

	this.renderer.clear([0.25, 0.5, 0.5, 1.0])
	// rotate 90 degrees around y
	this.renderer.sliceX(new Float32Array([
		0, 0, 1, 0,
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 0, 1]), new Float32Array([
		1.9, 0, 0, 0,
		0, 1.9, 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1]), this.isovalue_)
	this.renderer.blit(this.gl.drawingBufferWidth/2, this.gl.drawingBufferHeight/2, this.gl.drawingBufferWidth, this.gl.drawingBufferHeight)
}


ExampleViewer.prototype.uploadData = function(typedArray, width, height, depth, boxWidth, boxHeight, boxDepth) {
	this.currentDataSet = {
		'typedArray': typedArray,
		'width': width,
		'height': height,
		'depth': depth,
		'boxWidth': boxWidth,
		'boxHeight': boxHeight,
		'boxDepth': boxDepth,
	}
	this.renderer.uploadData(typedArray, width, height, depth, boxWidth, boxHeight, boxDepth)

	this.q = quat(1.0, 0.0, 0.0, 0.0)
	this.viewDistance = 1.5
	this.viewMatrix = new Float32Array([
		1.0, 0.0, 0.0, 0.0,
		0.0, 1.0, 0.0, 0.0,
		0.0, 0.0, 1.0, 0.0,
		0.0, 0.0, -this.viewDistance, 1.0,
	])

	this.render()
}


// quaternions
function quat(w, x, y, z)
{
	return {w, x, y, z}
}

function quatMul(q0, q1)
{
	return quat(q0.w*q1.w - q0.x*q1.x - q0.y*q1.y - q0.z*q1.z,
		q0.w*q1.x + q1.w*q0.x + q0.y*q1.z - q0.z*q1.y,
		q0.w*q1.y + q1.w*q0.y - q0.x*q1.z + q0.z*q1.x,
		q0.w*q1.z + q1.w*q0.z + q0.x*q1.y - q0.y*q1.x)
}

function
quatToMat4(q)
{
	return new Float32Array([
		1.0 - 2.0*q.y*q.y - 2.0*q.z*q.z, 2.0*q.x*q.y + 2.0*q.w*q.z, 2.0*q.x*q.z - 2.0*q.w*q.y, 0.0,
		2.0*q.x*q.y - 2.0*q.w*q.z, 1.0 - 2.0*q.x*q.x - 2.0*q.z*q.z, 2.0*q.y*q.z + 2.0*q.w*q.x, 0.0,
		2.0*q.x*q.z + 2.0*q.w*q.y, 2.0*q.y*q.z - 2.0*q.w*q.x, 1.0 - 2.0*q.x*q.x - 2.0*q.y*q.y, 0.0,
		0.0, 0.0, 0.0, 1.0,
	])
}



// Shoemake's arcball
function
arcball_screenToSphere(circle, screenX, screenY)
{
	const x = (screenX - circle.center.x)/circle.radius
	const y = -(screenY - circle.center.y)/circle.radius
	const r = x*x + y*y

	if (r > 1.0) {
		const s = 1.0/Math.sqrt(r)
		return {x: s*x, y: s*y, z: 0.0}
	} else {
		return {x: x, y: y, z: Math.sqrt(1.0 - r)}
	}
}

function
arcball_quat(startPoint, endPoint)
{
	function
	dot(u, v)
	{
		return u.x*v.x + u.y*v.y + u.z*v.z
	}

	function
	cross(u, v)
	{
		return {x: u.y*v.z - u.z*v.y, y: u.z*v.x - u.x*v.z, z: u.x*v.y - u.y*v.x}
	}

	const axis = cross(startPoint, endPoint)
	const angle = dot(startPoint, endPoint)
	return quat(angle, axis.x, axis.y, axis.z)
}
