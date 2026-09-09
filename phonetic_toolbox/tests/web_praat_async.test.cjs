const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const test = require("node:test");
const assert = require("node:assert/strict");

const source = fs.readFileSync(path.join(__dirname, "../gui/resources/web_praat_editor/app.js"), "utf8");

function setup() {
  const pending = new Map();
  const context = {
    AbortController, console,
    state: { activeId: null, loadSequence: 0, navigationSequence: 0, loadController: null, lipData: null, lipOffset: 0, lipDirty: false },
    els: { visible: { value: "3.2" } },
    fetch(url) {
      return new Promise((resolve, reject) => pending.set(url, { resolve, reject }));
    },
    showLipControls() {}, updateLipInfo() {}, drawSpectrogram() {},
    stopAudio() {}, renderFileList() {}, drawAll() {}, normalizeTextGrid() {},
    parseTextGrid: (text) => ({ text }),
    audioContext: () => ({ decodeAudioData: async (data) => data }),
    setStatus(message) { context.status = message; },
  };
  vm.createContext(context);
  vm.runInContext(source.slice(source.indexOf("async function loadItemSafely("), source.indexOf("function beginItemLoad(")), context);
  vm.runInContext(source.slice(source.indexOf("function beginItemLoad("), source.indexOf("function showLipControls(")), context);
  vm.runInContext(source.slice(source.indexOf("async function saveLipAlignment("), source.indexOf("function audioContext(")), context);
  const take = (prefix) => {
    const entry = [...pending].find(([url]) => url.startsWith(prefix));
    assert.ok(entry, `Missing request ${prefix}`);
    pending.delete(entry[0]);
    return entry[1];
  };
  return { context, take };
}

const flush = () => new Promise(setImmediate);
const response = (data) => ({ ok: true, json: async () => data });

test("a late lip response cannot overwrite another recording", async () => {
  const { context: c, take } = setup();
  c.state.activeId = "A";
  const a = c.loadLipData("A", c.beginItemLoad());
  const old = take("/api/lip?id=A");
  c.state.activeId = "B";
  const b = c.loadLipData("B", c.beginItemLoad());
  take("/api/lip?id=B").resolve(response({ available: true, offset: 0.2, marker: "B" }));
  await b;
  old.resolve(response({ available: true, offset: 0.7, marker: "A" }));
  await a;
  assert.equal(c.state.lipData.marker, "B");
  assert.equal(c.state.lipOffset, 0.2);
});

test("a late failure cannot clear current lip edits", async () => {
  const { context: c, take } = setup();
  c.state.activeId = "A";
  const a = c.loadLipData("A", c.beginItemLoad());
  const old = take("/api/lip?id=A");
  c.beginItemLoad();
  c.state.activeId = "B";
  c.state.lipData = { marker: "B" };
  c.state.lipDirty = true;
  old.reject(new Error("old request failed"));
  await a;
  assert.equal(c.state.lipData.marker, "B");
  assert.equal(c.state.lipDirty, true);
});

test("a rescan invalidates lip loads even before the next file opens", async () => {
  const { context: c, take } = setup();
  c.state.activeId = "A";
  const a = c.loadLipData("A", c.beginItemLoad());
  const old = take("/api/lip?id=A");
  c.beginItemLoad();
  old.resolve(response({ available: true, offset: 0.7 }));
  await a;
  assert.equal(c.state.lipData, null);
});

test("late audio decoding cannot combine A audio with B annotations", async () => {
  const { context: c, take } = setup();
  const a = c.loadItem("A");
  take("/api/item?id=A").resolve(response({ id: "A", textgrid: "A", audioUrl: "/audioA" }));
  await flush();
  const old = take("/audioA");
  const b = c.loadItem("B");
  take("/api/item?id=B").resolve(response({ id: "B", textgrid: "B", audioUrl: "/audioB" }));
  await flush();
  take("/audioB").resolve({ ok: true, arrayBuffer: async () => ({ duration: 2, marker: "B" }) });
  await b;
  old.resolve({ ok: true, arrayBuffer: async () => ({ duration: 1, marker: "A" }) });
  await a;
  assert.equal(c.state.activeId, "B");
  assert.equal(c.state.textgrid.text, "B");
  assert.equal(c.state.audioBuffer.marker, "B");
});

test("a save response preserves edits made while the request was running", async () => {
  const { context: c, take } = setup();
  c.state.activeId = "A";
  c.state.lipData = { offset: 0 };
  c.state.lipOffset = 0.1;
  const saving = c.saveLipAlignment({ silent: true });
  c.state.lipOffset = 0.2;
  c.state.lipDirty = true;
  take("/api/lip/save").resolve(response({ ok: true }));
  await saving;
  assert.equal(c.state.lipDirty, true);
  assert.equal(c.state.lipOffset, 0.2);
  assert.equal(c.state.lipData.offset, 0.1);
});

test("late save completion cannot reverse the latest navigation choice", async () => {
  const { context: c } = setup();
  const pendingSaves = [];
  const opened = [];
  c.savePendingChanges = () => new Promise((resolve) => pendingSaves.push(resolve));
  c.loadItem = async (id) => opened.push(id);
  const a = c.loadItemSafely("A");
  const b = c.loadItemSafely("B");
  pendingSaves[1](true);
  await b;
  pendingSaves[0](true);
  await a;
  assert.deepEqual(opened, ["B"]);
});
