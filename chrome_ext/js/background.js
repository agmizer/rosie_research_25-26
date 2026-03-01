chrome.action.onClicked.addListener(() => {
  chrome.windows.create({
    url: "../html/window.html",
    type: "popup",
    width: 900,
    height: 700
  });
});