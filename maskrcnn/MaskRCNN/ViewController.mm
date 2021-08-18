//
//  ViewController.m
//  MaskRCNN
//
//  Created by Tao Xu on 8/13/21.
//

#import "ViewController.h"
#import "MaskRCNNModelRunner.h"
#import <LibTorch/LibTorch.h>
@interface ViewController ()

@end

@implementation ViewController{
    UIImageView *_imageView;
    MaskRCNNModelRunner* _modelRunner;
}

- (void)viewDidLoad {
    [super viewDidLoad];
    self.view.backgroundColor = [UIColor whiteColor];
    self.title = @"MaskRCNN";
    self.edgesForExtendedLayout = UIRectEdgeNone;
    _imageView = [[UIImageView alloc] initWithFrame:self.view.bounds];
    _imageView.contentMode = UIViewContentModeScaleAspectFit;
    [self.view addSubview:_imageView];
    _modelRunner = [MaskRCNNModelRunner new];
    [self showSpinner];
    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
        [self->_modelRunner loadModel];
        // warmup
        UIImage* image = [self->_modelRunner run];
        // benchmark
        image = [self->_modelRunner run];
        dispatch_async(dispatch_get_main_queue(), ^{
            [self hideSpinner];
            self->_imageView.image = image;
        });
    });
}


- (void)showSpinner
{
    UIView *view = [[UIView alloc] initWithFrame:CGRectMake(0, 0, 80, 80)];
    view.tag = 99;
    view.translatesAutoresizingMaskIntoConstraints = NO;
    view.layer.cornerRadius = 8;
    view.layer.masksToBounds = YES;
    view.backgroundColor = [UIColor grayColor];
    UIActivityIndicatorView *spinner = [[UIActivityIndicatorView alloc]
                                        initWithActivityIndicatorStyle:UIActivityIndicatorViewStyleWhiteLarge];
    CGRect frame = CGRectMake(0, 0, 40, 40);
    frame.origin.x = view.frame.size.width / 2 - frame.size.width / 2;
    frame.origin.y = view.frame.size.height / 2 - frame.size.height / 2;
    spinner.frame = frame;
    [view addSubview:spinner];
    [spinner startAnimating];
    view.center = self.view.center;
    [self.view addSubview:view];
}

- (void)hideSpinner
{
    [[self.view viewWithTag:99] removeFromSuperview];
}


@end
