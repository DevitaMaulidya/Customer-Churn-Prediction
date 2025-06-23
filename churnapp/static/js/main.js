$(window).on("load", function () {
  if ($("#churnTable").length) {
    $("#churnTable").DataTable({
      deferRender: true,
      processing: true,
      pageLength: 10,
      order: [[0, "asc"]],
    });
  }

  if ($("#churnTable1").length) {
    $("#churnTable1").DataTable({
      deferRender: true,
      processing: true,
      pageLength: 10,
      order: [[0, "asc"]],
    });
  }

  if ($("#churnTable2").length) {
    $("#churnTable2").DataTable({
      deferRender: true,
      processing: true,
      pageLength: 10,
      order: [[0, "asc"]],
    });
  }

  if ($("#churnTable3").length) {
    $("#churnTable3").DataTable({
      deferRender: true,
      processing: true,
      pageLength: 10,
      order: [[0, "asc"]],
    });
  }
});
