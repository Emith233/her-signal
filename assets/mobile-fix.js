(function() {
  var embedded = false;
  try {
    embedded = new URLSearchParams(window.location.search).get('embedded') === '1';
  } catch (e) {}
  if (!embedded) return;

  document.documentElement.classList.add('embedded');
  function markBody() {
    if (document.body) document.body.classList.add('embedded');
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', markBody, { once: true });
  } else {
    markBody();
  }
})();
