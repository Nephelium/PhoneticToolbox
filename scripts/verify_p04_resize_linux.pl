#!/usr/bin/env perl
# P04-RESIZE: Linux filesystem / bundle checks only, no GUI or server claims.
use strict;
use warnings;
use Cwd qw(abs_path);
use Digest::SHA qw(sha256_hex);
use File::Basename qw(dirname);
use File::Find;
use File::Path qw(make_path);
use JSON::PP;

my $root = abs_path(dirname(__FILE__) . '/..');
my $dist = "$root/frontend/dist";
die "missing production bundle\n" unless -f "$dist/index.html";
my @paths;
find({no_chdir => 1, wanted => sub {
    push @paths, $File::Find::name if -f $_ && /\.(?:js|css|html|ttf|woff2)$/;
}}, $dist);
my @files;
my $shared = 0;
for my $path (sort @paths) {
    open my $fh, '<:raw', $path or die $!;
    local $/; my $body = <$fh>; close $fh;
    $shared++ if $path =~ /\.js$/ && index($body, 'panel-resize-handle') >= 0;
    push @files, {path => substr($path, length($dist) + 1), bytes => length($body), sha256 => sha256_hex($body)};
}
die "missing shared boundary controller\n" unless $shared;
open my $html, '<:raw', "$dist/vocal-tract/index.html" or die $!;
{ local $/; my $body = <$html>; die "old M10 heading or close action\n" if $body =~ /id="shutdownButton"|<h1>/; }
close $html;
my $out = "$root/output/validation/p04-resize/linux-static-" . time;
make_path($out);
open my $report, '>:encoding(UTF-8)', "$out/report.json" or die $!;
print {$report} JSON::PP->new->pretty->canonical->encode({
    success => JSON::PP::true, platform => scalar(`uname -srm`) =~ s/\s+\z//r,
    scope => 'WSL Linux read and SHA-256 of Windows production bundle; no Linux browser/Qt/server test',
    files => \@files,
});
close $report;
print "$out/report.json\n";
