
$(document).ready(function(){
		$('.ml-algo').click(function(){
			var id = $(this).attr("id");
			id = "#" + id + "-graph";
			$('.ml-graph').hide();
			$(id).show();
		});
	
});